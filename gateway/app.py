import os
import asyncio
from typing import Optional, Any, Dict
import json
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
    Expects:
      {
        "prompt": "…",
        "model": "mistral:7b-instruct-q4_K_M",     # optional (есть дефолт)
        "system": "Ты — русскоязычный ассистент",  # optional (есть дефолт)
        "options": {"temperature":0.2, ...},       # optional
        "stream": false                             # optional
      }
    Forwards to Ollama /api/generate and tolerates both streaming and non-streaming.
    """
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(400, "Field 'prompt' is required")

    model = payload.get("model") or "mistral:7b-instruct-q4_K_M"
    if model in ["qwen", "qwen25", "qwen-ru"]:
        model = "qwen2.5:7b-instruct-q4_K_M"
    system = payload.get("system") or "Ты — русскоязычный ассистент. Всегда отвечай по-русски, кратко и грамотно."
    options = payload.get("options") or {"temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.1}
    stream = payload.get("stream")
    if stream is None:
        stream = False  # по умолчанию работаем в нестримовом режиме

    req = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "options": options,
        "stream": stream
    }

    # 1) сначала пробуем обычный (non-streaming) путь
    if stream is False:
        r = await client.post(f"{OLLAMA}/api/generate", json=req)
        if r.is_error:
            raise HTTPException(r.status_code, r.text)

        # Иногда Ollama может вернуть несколько JSON'ов даже при stream=false (редко).
        # Попробуем безопасный парсинг: если есть несколько строк JSON — склеим response.
        txt = r.text.strip()
        try:
            data = json.loads(txt)
            return JSONResponse(data)
        except json.JSONDecodeError:
            # fallback: соберём построчно
            response_text = ""
            done_obj = {}
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    response_text += obj.get("response", "")
                    if obj.get("done"):
                        done_obj = obj
                except Exception:
                    # игнорируем мусорные строки
                    continue
            if not response_text and not done_obj:
                # совсем не смогли распарсить — вернём как есть
                return JSONResponse({"raw": txt})
            merged = {"model": model, "response": response_text, "done": True}
            if done_obj:
                merged.update({k: v for k, v in done_obj.items() if k not in merged})
            return JSONResponse(merged)

    # 2) stream=true: читаем по строкам и агрегируем
    async with client.stream("POST", f"{OLLAMA}/api/generate", json=req) as resp:
        if resp.is_error:
            text = await resp.aread()
            raise HTTPException(resp.status_code, text.decode("utf-8", errors="ignore"))
        response_text = ""
        done_obj = {}
        async for chunk in resp.aiter_lines():
            if not chunk:
                continue
            try:
                obj = json.loads(chunk)
                response_text += obj.get("response", "")
                if obj.get("done"):
                    done_obj = obj
            except Exception:
                continue
        merged = {"model": model, "response": response_text, "done": True}
        if done_obj:
            merged.update({k: v for k, v in done_obj.items() if k not in merged})
        return JSONResponse(merged)

# ---------------------------
# STT (Whisper)  /stt
# ---------------------------
@app.post("/stt")
async def stt(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    files = {"file": (file.filename, await file.read(), file.content_type or "application/octet-stream")}
    data = {}
    if language:
        data["language"] = language

    # небольшой ретрай на случай первого старта модели/инициализации CUDA
    for attempt in range(2):
        try:
            r = await client.post(f"{WHISPER}/transcribe", files=files, data=data)
            if r.is_error:
                raise HTTPException(r.status_code, r.text)
            return JSONResponse(r.json())
        except httpx.RemoteProtocolError as e:
            if attempt == 0:
                await asyncio.sleep(1.0)
                continue
            raise HTTPException(status_code=502, detail=f"whisper upstream disconnected: {e}")

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
