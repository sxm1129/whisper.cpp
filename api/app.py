"""
Whisper HTTP API based on whisper.cpp

Expose simple /health and /transcribe endpoints on top of the
CUDA-accelerated whisper.cpp CLI binary in this repo.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

BASE_DIR = Path(__file__).resolve().parents[1]

# Default binary and model locations inside this repo
DEFAULT_WHISPER_BIN = str(BASE_DIR / "build" / "bin" / "whisper-cli")
DEFAULT_MODELS_DIR = str(BASE_DIR / "models")

WHISPER_BIN = os.getenv("WHISPER_BIN", DEFAULT_WHISPER_BIN)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
WHISPER_MODELS_DIR = os.getenv("WHISPER_MODELS_DIR", DEFAULT_MODELS_DIR)

app = FastAPI(title="whisper.cpp HTTP API", version="1.0.0")


def find_model_path() -> Optional[str]:
    """
    Locate ggml model file based on WHISPER_MODEL and common search locations.
    """
    model_name = f"ggml-{WHISPER_MODEL}.bin"

    search_dirs = [
        Path(WHISPER_MODELS_DIR),
        BASE_DIR / "models",
        Path.home() / ".cache" / "whisper",
    ]

    for d in search_dirs:
        path = d / model_name
        if path.exists():
            return str(path)
    return None


@app.get("/health")
async def health() -> Dict[str, Any]:
    """
    Basic health check.
    """
    whisper_found = shutil.which(WHISPER_BIN) is not None or Path(WHISPER_BIN).exists()
    model_path = find_model_path()

    return {
        "status": "ok",
        "whisper_binary": WHISPER_BIN,
        "whisper_found": whisper_found,
        "model": WHISPER_MODEL,
        "model_found": model_path is not None,
        "model_path": model_path,
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="Audio file (WAV/MP3/...)"),
    language: str = Form("zh", description="Language code: zh/en/auto"),
    word_timestamps: bool = Form(True, description="Return word-level timestamps"),
) -> Dict[str, Any]:
    """
    Transcribe an audio file using whisper.cpp and return caption-style output.
    """
    # Check whisper binary
    binary_path = None
    if Path(WHISPER_BIN).exists():
        binary_path = WHISPER_BIN
    else:
        which_res = shutil.which(WHISPER_BIN)
        if which_res:
            binary_path = which_res
    if not binary_path:
        raise HTTPException(status_code=503, detail=f"whisper.cpp binary not found: {WHISPER_BIN}")

    # Check model
    model_path = find_model_path()
    if not model_path:
        raise HTTPException(
            status_code=503,
            detail=f"Whisper model not found: ggml-{WHISPER_MODEL}.bin (searched in {WHISPER_MODELS_DIR} and defaults)",
        )

    # Save uploaded audio to temp file
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        while True:
            chunk = await audio.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)

    wav_path = tmp_path
    try:
        # Convert to 16kHz mono WAV if needed
        if suffix.lower() != ".wav":
            wav_path = tmp_path + ".wav"
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    tmp_path,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-f",
                    "wav",
                    wav_path,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode != 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio conversion failed: {proc.stderr[:200]}",
                )

        # Build whisper.cpp command
        cmd = [
            binary_path,
            "-m",
            model_path,
            "-f",
            wav_path,
            "-l",
            language if language != "auto" else "auto",
            "--output-json-full",
            "--no-prints",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Whisper transcription failed: {proc.stderr[:300]}",
            )

        # whisper.cpp --output-json-full writes <wav_path>.json
        json_path = wav_path + ".json"
        json_file = Path(json_path)
        if not json_file.exists():
            alt = Path(wav_path).with_suffix(".json")
            if alt.exists():
                json_file = alt
            else:
                raise HTTPException(status_code=500, detail="Whisper did not produce JSON output")

        with json_file.open("r", encoding="utf-8", errors="replace") as f:
            whisper_data = json.load(f)

        captions = _parse_whisper_output(whisper_data, word_timestamps)
        full_text = " ".join(c["text"] for c in captions)

        return {
            "success": True,
            "captions": captions,
            "full_text": full_text,
        }
    finally:
        # Cleanup temp files
        for p in {tmp_path, tmp_path + ".wav", tmp_path + ".json", tmp_path + ".wav.json"}:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


def _parse_whisper_output(data: Dict[str, Any], word_level: bool) -> List[Dict[str, Any]]:
    """
    Convert whisper.cpp JSON output (from --output-json-full) into
    a list of caption dicts with ms timestamps.
    """
    captions: List[Dict[str, Any]] = []

    for segment in data.get("transcription", []):
        if word_level and "tokens" in segment:
            for token in segment["tokens"]:
                text = token.get("text", "").strip()
                if not text or text.startswith("["):
                    continue
                offsets = token.get("offsets", {})
                captions.append(
                    {
                        "text": text,
                        "startMs": int(offsets.get("from", 0)),
                        "endMs": int(offsets.get("to", 0)),
                        "confidence": float(token.get("p", 0.0)),
                    }
                )
        else:
            text = segment.get("text", "").strip()
            if not text:
                continue
            ts = segment.get("timestamps", {})
            start_ms = _time_to_ms(ts.get("from", "00:00:00.000"))
            end_ms = _time_to_ms(ts.get("to", "00:00:00.000"))
            captions.append(
                {
                    "text": text,
                    "startMs": start_ms,
                    "endMs": end_ms,
                    "confidence": 1.0,
                }
            )

    return captions


def _time_to_ms(time_str: str) -> int:
    """
    Convert whisper.cpp time string "HH:MM:SS.mmm" to milliseconds.
    """
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

