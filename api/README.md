# Whisper HTTP API (whisper.cpp)

This directory provides a lightweight HTTP transcription API on top of the
`whisper.cpp` CLI, intended to run on the same machine as the compiled
CUDA-accelerated binaries.

## Prerequisites

- CUDA and NVIDIA drivers properly installed (L20 GPU in your case)
- System packages:
  - `build-essential`, `cmake`, `git`, `ffmpeg`, `python3-pip`
- Conda environment `whispercpp_env` with:
  - `fastapi`, `uvicorn[standard]`, `python-multipart`

## Build whisper.cpp (once)

From the project root:

```bash
cd /data1/workspace/whisper.cpp
cmake -B build -DGGML_CUDA=1
cmake --build build -j4 --config Release
```

This produces the CLI binary at:

- `./build/bin/whisper-cli`

## Model files

Download a `ggml` model (for example `medium`) into the `models/` folder:

```bash
cd /data1/workspace/whisper.cpp
bash models/download-ggml-model.sh medium
```

If direct download is not possible due to network restrictions, place
`ggml-medium.bin` manually into:

- `/data1/workspace/whisper.cpp/models/ggml-medium.bin`

## Python environment

Create and activate the Conda environment:

```bash
conda create -n whispercpp_env python=3.10 -y
conda activate whispercpp_env
pip install -r api/requirements.txt
```

## Running the HTTP server

From the project root:

```bash
cd /data1/workspace/whisper.cpp
conda activate whispercpp_env

# Optional overrides:
export WHISPER_BIN="./build/bin/whisper-cli"
export WHISPER_MODEL="medium"
export WHISPER_MODELS_DIR="./models"

uvicorn api.app:app --host 0.0.0.0 --port 8700
```

## Endpoints

- `GET /health`
  - Returns basic status, binary and model availability flags.

- `POST /transcribe`
  - `multipart/form-data`
    - `audio`: audio file (wav/mp3, etc.)
    - `language`: `zh` / `en` / `auto` (default: `zh`)
    - `word_timestamps`: boolean, whether to return word-level timestamps (default: `true`)

