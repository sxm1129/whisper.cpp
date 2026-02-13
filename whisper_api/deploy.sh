#!/bin/bash
# ============================================
# Whisper API 一键部署脚本 (L20 GPU 服务器)
# 使用方法:
#   1. 将 services/whisper_api/ 目录上传到 L20 服务器
#   2. chmod +x deploy.sh && ./deploy.sh
# ============================================

set -e

echo "🚀 Whisper API 部署脚本 (L20 GPU)"
echo "=================================="

# ─── 1. 检查 CUDA ────────────────────────
echo ""
echo "📋 步骤 1/5: 检查 CUDA 环境..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✅ CUDA 可用"
else
    echo "⚠️  nvidia-smi 未找到，whisper.cpp 将以 CPU 模式运行"
fi

# ─── 2. 安装系统依赖 ──────────────────────
echo ""
echo "📋 步骤 2/5: 安装系统依赖..."
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential cmake git ffmpeg python3-pip python3-venv

# ─── 3. 编译 whisper.cpp (GPU 加速) ───────
echo ""
echo "📋 步骤 3/5: 编译 whisper.cpp..."
WHISPER_DIR="$HOME/whisper.cpp"

if [ ! -d "$WHISPER_DIR" ]; then
    git clone https://github.com/ggerganov/whisper.cpp.git "$WHISPER_DIR"
fi

cd "$WHISPER_DIR"
git pull

# 检测是否有 CUDA，选择编译方式
if command -v nvidia-smi &> /dev/null; then
    echo "🔥 使用 CUDA 编译 (GPU 加速)..."
    cmake -B build -DGGML_CUDA=ON
else
    echo "📦 使用 CPU 编译..."
    cmake -B build
fi

cmake --build build -j$(nproc)

# 安装到 /usr/local/bin
sudo cp build/bin/whisper-cli /usr/local/bin/whisper-cpp 2>/dev/null || \
sudo cp build/bin/main /usr/local/bin/whisper-cpp 2>/dev/null || \
echo "⚠️ 找不到编译产物，请检查 build/bin/ 目录"

echo "✅ whisper.cpp 编译完成"
whisper-cpp --help 2>&1 | head -3

# ─── 4. 下载模型 ──────────────────────────
echo ""
echo "📋 步骤 4/5: 下载 Whisper 模型 (medium)..."
MODEL_DIR="$HOME/.cache/whisper"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/ggml-medium.bin" ]; then
    echo "📥 下载 ggml-medium.bin (约 1.5GB)..."
    cd "$WHISPER_DIR"
    bash models/download-ggml-model.sh medium
    cp models/ggml-medium.bin "$MODEL_DIR/"
    echo "✅ 模型下载完成"
else
    echo "✅ 模型已存在，跳过下载"
fi

# ─── 5. 启动 API 服务 ─────────────────────
echo ""
echo "📋 步骤 5/5: 启动 Whisper API 服务..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 创建虚拟环境
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt

echo ""
echo "🎉 部署完成！启动命令："
echo ""
echo "  # 前台运行（测试用）:"
echo "  cd $SCRIPT_DIR && source venv/bin/activate"
echo "  python app.py"
echo ""
echo "  # 后台运行（生产用）:"
echo "  cd $SCRIPT_DIR && source venv/bin/activate"
echo "  nohup uvicorn app:app --host 0.0.0.0 --port 8700 --workers 2 > whisper.log 2>&1 &"
echo ""
echo "  # 健康检查:"
echo "  curl http://localhost:8700/health"
echo ""
