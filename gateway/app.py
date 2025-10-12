import os

import httpx
from fastapi import FastAPI

app = FastAPI(
    title="Local AI Aggregator Gateway",
    description="Шлюз к локальным сервисам: Ollama (LLM), Faster-Whisper (STT), XTTS (TTS), ComfyUI (image).",
)

OLLAMA = os.getenv("OLLAMA_URL", "http://ollama:11434")
XTTS   = os.getenv("XTTS_URL",   "http://xtts:8021")
WHISPER= os.getenv("WHISPER_URL","http://whisper:8022")
COMFY  = os.getenv("COMFY_URL",  "http://comfyui:8188")

TIMEOUT = httpx.Timeout(120.0, connect=10.0, read=120.0)
client = httpx.AsyncClient(timeout=TIMEOUT)

@app.on_event("shutdown")
async def _shutdown():
    await client.aclose()

@app.get("/health", summary="Проверка здоровья", tags=["Service"])
async def health():
    return {"status": "ok"}
