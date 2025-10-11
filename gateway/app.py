import os
import asyncio
from typing import Optional, Any, Dict

import httpx
from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, Response

app = FastAPI(title="Local AI Aggregator Gateway")

# --- targets (internal docker hostnames from compose) ---
OLLAMA = os.getenv("OLLAMA_URL", "http://ollama:11434")
XTTS   = os.getenv("XTTS_URL",   "http://xtts:8021")
WHISPER= os.getenv("WHISPER_URL","http://whisper:8022")
COMFY  = os.getenv("COMFY_URL",  "http://comfyui:8188")

TIMEOUT = httpx.Timeout(120.0, connect=10.0, read=120.0)
client = httpx.AsyncClient(timeout=TIMEOUT)

@app.on_event("shutdown")
async def _shutdown():
    await client.aclose()

@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------------------------
# LLM (Ollama)  /chat
# ---------------------------
@app.post("/chat")
async def chat(payload: Dict[str, Any] = Body(...)):
    """
    Expects: {"prompt": "...", "model": "mistral:7b-instruct-q4_K_M", ...ollama options}
    Forwards to: POST {OLLAMA}/api/generate
    """
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(400, "Field 'prompt' is required")

    # set default model if not provided
    payload.setdefault("model", "mistral:7b-instruct-q4_K_M")
    # ensure we don't accidentally enable streaming in gateway
    payload.setdefault("stream", False)

    r = await client.post(f"{OLLAMA}/api/generate", json=payload)
    if r.is_error:
        raise HTTPException(r.status_code, r.text)
    return JSONResponse(r.json())

# ---------------------------
# STT (Whisper)  /stt
# ---------------------------
@app.post("/stt")
async def stt(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    """
    Forwards multipart to: POST {WHISPER}/transcribe
    """
    files = {"file": (file.filename, await file.read(), file.content_type or "application/octet-stream")}
    data = {}
    if language:
        data["language"] = language

    r = await client.post(f"{WHISPER}/transcribe", files=files, data=data)
    if r.is_error:
        raise HTTPException(r.status_code, r.text)
    return JSONResponse(r.json())

# ---------------------------
# TTS (XTTS)  /tts
# ---------------------------
@app.post("/tts")
async def tts(body: Dict[str, Any] = Body(...)):
    """
    Expects: {"text":"...", "language":"ru", "speaker_wav": null}
    Forwards JSON to: POST {XTTS}/tts
    Returns audio/wav stream to the client.
    """
    if "text" not in body:
        raise HTTPException(400, "Field 'text' is required")

    # Request TTS
    r = await client.post(f"{XTTS}/tts", json=body)
    if r.is_error:
        raise HTTPException(r.status_code, r.text)

    # Stream audio back
    media = r.headers.get("content-type", "audio/wav")
    return Response(content=r.content, media_type=media, headers={
        "Content-Disposition": 'inline; filename="speech.wav"'
    })

# ---------------------------
# IMAGE (ComfyUI)  /image/raw
# ---------------------------
@app.post("/image/raw")
async def comfy_raw(payload: Dict[str, Any] = Body(...)):
    """
    Transparent proxy to ComfyUI prompt endpoint.
    Expects a ready ComfyUI payload and forwards to:
      POST {COMFY}/prompt
    Returns ComfyUI JSON response as-is.
    """
    r = await client.post(f"{COMFY}/prompt", json=payload)
    if r.is_error:
        raise HTTPException(r.status_code, r.text)
    return JSONResponse(r.json())
