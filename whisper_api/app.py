"""
Whisper API 服务 — 部署到 L20 GPU 服务器
提供 HTTP 接口供本地 caption_service 远程调用

启动: uvicorn app:app --host 0.0.0.0 --port 8700
"""

import os
import json
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-api")

app = FastAPI(title="Whisper Transcription API", version="1.0.0")

# ─── 配置 ────────────────────────────────────────────
WHISPER_BIN = os.getenv("WHISPER_BIN", "whisper-cpp")  # whisper.cpp 可执行文件路径
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")     # 模型名称
WHISPER_MODELS_DIR = os.getenv("WHISPER_MODELS_DIR", str(Path.home() / ".cache" / "whisper"))


def find_model_path() -> Optional[str]:
    """查找 whisper 模型文件"""
    model_name = f"ggml-{WHISPER_MODEL}.bin"
    search_dirs = [
        Path(WHISPER_MODELS_DIR),
        Path.home() / ".cache" / "whisper",
        Path.home() / "whisper.cpp" / "models",
        Path("/usr/local/share/whisper"),
    ]
    for d in search_dirs:
        path = d / model_name
        if path.exists():
            return str(path)
    return None


@app.get("/health")
async def health():
    """健康检查"""
    model_path = find_model_path()
    whisper_found = shutil.which(WHISPER_BIN) is not None
    return {
        "status": "ok",
        "whisper_binary": WHISPER_BIN,
        "whisper_found": whisper_found,
        "model": WHISPER_MODEL,
        "model_found": model_path is not None,
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="音频文件 (WAV/MP3)"),
    language: str = Form("zh", description="语言代码: zh/en/auto"),
    word_timestamps: bool = Form(True, description="是否返回逐字时间戳"),
):
    """
    转录音频文件，返回逐字时间戳

    Returns:
        {
            "success": true,
            "captions": [
                {"text": "你好", "startMs": 100, "endMs": 500, "confidence": 0.95},
                ...
            ],
            "full_text": "你好世界"
        }
    """
    # 校验 whisper 可用性
    whisper_bin = shutil.which(WHISPER_BIN)
    if not whisper_bin:
        raise HTTPException(status_code=503, detail=f"whisper.cpp 未找到: {WHISPER_BIN}")

    model_path = find_model_path()
    if not model_path:
        raise HTTPException(status_code=503, detail=f"Whisper 模型未找到: ggml-{WHISPER_MODEL}.bin")

    # 保存上传的音频到临时文件
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        while True:
            chunk = await audio.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)

    try:
        # 如果不是 WAV 16KHz，先转换
        wav_path = tmp_path
        if suffix.lower() != ".wav":
            wav_path = tmp_path + ".wav"
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode != 0:
                raise HTTPException(status_code=400, detail=f"音频转换失败: {proc.stderr[:200]}")

        # 构建 whisper.cpp 命令
        cmd = [
            whisper_bin,
            "-m", model_path,
            "-f", wav_path,
            "-l", language if language != "auto" else "auto",
            "--output-json-full",  # 输出完整 JSON（含 token 时间戳）
            "--no-prints",         # 减少 stderr 输出
        ]

        logger.info(f"执行: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Whisper 转录失败: {proc.stderr[:300]}")

        # whisper.cpp --output-json-full 会生成 {wav_path}.json
        json_path = wav_path + ".json"
        if not Path(json_path).exists():
            # 尝试不带 .wav 后缀
            alt_json = Path(wav_path).with_suffix(".json")
            if alt_json.exists():
                json_path = str(alt_json)
            else:
                raise HTTPException(status_code=500, detail="Whisper 未生成 JSON 输出")

        with open(json_path, "r", encoding="utf-8") as f:
            whisper_data = json.load(f)

        # 解析为标准 Caption 格式
        captions = _parse_whisper_output(whisper_data, word_timestamps)

        full_text = " ".join(c["text"] for c in captions)
        logger.info(f"转录完成: {len(captions)} 段, {len(full_text)} 字")

        return {
            "success": True,
            "captions": captions,
            "full_text": full_text,
        }

    finally:
        # 清理临时文件
        for f in [tmp_path, tmp_path + ".wav", tmp_path + ".json", tmp_path + ".wav.json"]:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass


def _parse_whisper_output(data: dict, word_level: bool) -> list:
    """解析 whisper.cpp JSON 输出为 Caption[] 格式"""
    captions = []

    for segment in data.get("transcription", []):
        if word_level and "tokens" in segment:
            # token 级别（逐字）时间戳
            for token in segment["tokens"]:
                text = token.get("text", "").strip()
                if not text or text.startswith("["):
                    continue
                captions.append({
                    "text": text,
                    "startMs": int(token.get("offsets", {}).get("from", 0)),
                    "endMs": int(token.get("offsets", {}).get("to", 0)),
                    "confidence": token.get("p", 0.0),
                })
        else:
            # segment 级别时间戳
            text = segment.get("text", "").strip()
            if not text:
                continue

            # whisper.cpp 时间格式: "00:00:01.234" → 毫秒
            start_ms = _time_to_ms(segment.get("timestamps", {}).get("from", "00:00:00.000"))
            end_ms = _time_to_ms(segment.get("timestamps", {}).get("to", "00:00:00.000"))

            captions.append({
                "text": text,
                "startMs": start_ms,
                "endMs": end_ms,
                "confidence": 1.0,
            })

    return captions


def _time_to_ms(time_str: str) -> int:
    """将 whisper.cpp 时间格式 '00:00:01.234' 转为毫秒"""
    try:
        parts = time_str.split(":")
        h, m = int(parts[0]), int(parts[1])
        s_parts = parts[2].split(".")
        s = int(s_parts[0])
        ms = int(s_parts[1]) if len(s_parts) > 1 else 0
        return (h * 3600 + m * 60 + s) * 1000 + ms
    except Exception:
        return 0


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8700)
