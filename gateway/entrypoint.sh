#!/usr/bin/env bash
set -e
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
python3 - << 'PY'
import os
from faster_whisper import WhisperModel
name = os.getenv("WHISPER_MODEL","small")
compute = os.getenv("WHISPER_COMPUTE","int8_float16")
try:
    WhisperModel(name, device="cpu", compute_type="int8")
    print(f"Whisper cached ({name})")
except Exception as e:
    print("Warmup error:", e)
PY
exec uvicorn app:app --host 0.0.0.0 --port 8022
