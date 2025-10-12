import os
import asyncio
from os import getenv
from typing import Optional, Any, Dict, List, Literal
import json
import httpx
from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, ConfigDict

app = FastAPI(
    title="Local AI Aggregator Gateway",
    description="Шлюз к локальным сервисам: Ollama (LLM), Faster-Whisper (STT), XTTS (TTS), ComfyUI (image).",
)

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

@app.get("/health", summary="Проверка здоровья", tags=["Service"])
async def health():
    return {"status": "ok"}

# =========================
# Pydantic models (OpenAPI)
# =========================

Role = Literal["system", "user", "assistant"]

class ChatMessage(BaseModel):
    role: Role = Field(..., examples=["user"])
    content: str = Field(..., examples=["Привет! Кто ты?"])

class ChatOptions(BaseModel):
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="Креативность")
    num_ctx: Optional[int] = Field(2048, ge=256, description="Контекст (токены)")
    num_predict: Optional[int] = Field(256, ge=256, description="Длинна вывода")
    top_p: Optional[float] = Field(None, ge=0, le=1)
    top_k: Optional[int] = Field(None, ge=0)
    repeat_penalty: Optional[float] = Field(None, ge=0)

class ChatRequest(BaseModel):
    model: Optional[str] = Field("qwen2.5:3b-instruct-q4_K_M", description="Ollama model tag")
    messages: List[ChatMessage] = Field(..., description="История диалога")
    stream: Optional[bool] = Field(False, description="Стриминговый ответ")
    options: Optional[ChatOptions] = Field(
        default_factory=ChatOptions,
        description="Параметры генерации"
    )

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "model": "qwen2.5:3b-instruct-q4_K_M",
                "messages": [
                    {"role": "system", "content": "Ты — русскоязычный ассистент. Отвечай кратко."},
                    {"role": "user", "content": "Сформулируй правило трёх в одном предложении."}
                ],
                "stream": False,
                "options": {"temperature": 0.5, "num_ctx": 2048, "num_predict": 256}
            }
        ]
    })

# Разрешаем «лишние» поля, которые может вернуть Ollama
class ChatGatewayResponse(BaseModel):
    model: Optional[str] = None
    done: Optional[bool] = None
    message: Optional[ChatMessage] = None
    # Доп. поля от Ollama (total_duration, load_duration, eval_count, context, и т.п.)
    model_config = ConfigDict(extra="allow")

class MessageOptions(BaseModel):
    temperature: Optional[float] = Field(0.2, ge=0, le=2)
    top_p: Optional[float] = Field(0.9, ge=0, le=1)
    repeat_penalty: Optional[float] = Field(1.1, ge=0)

class MessageRequest(BaseModel):
    prompt: str = Field(..., description="Текст запроса")
    model: Optional[str] = Field("qwen2.5:3b-instruct-q4_K_M")
    system: Optional[str] = Field("Ты — русскоязычный ассистент. Всегда отвечай по-русски, кратко и грамотно.")
    options: Optional[MessageOptions] = Field(default_factory=MessageOptions)
    stream: Optional[bool] = Field(False)

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "prompt": "Суммируй: 1) Сборка; 2) Тест; 3) Деплой — в одном абзаце.",
                "model": "qwen2.5:3b-instruct-q4_K_M",
                "system": "Ты — русскоязычный ассистент. Отвечай кратко.",
                "options": {"temperature": 0.3, "top_p": 0.9, "repeat_penalty": 1.1},
                "stream": False
            }
        ]
    })

class GenerateGatewayResponse(BaseModel):
    model: Optional[str] = None
    response: Optional[str] = None
    done: Optional[bool] = None
    # Разрешаем все остальные поля от Ollama (/api/generate)
    model_config = ConfigDict(extra="allow")

class STTSegment(BaseModel):
    start: float
    end: float
    text: str

class STTResponse(BaseModel):
    language: Optional[str] = Field(None, description="Детектированный язык (если есть)")
    text: str = Field(..., description="Итоговая транскрипция")
    segments: Optional[List[STTSegment]] = Field(None, description="Сегменты (если возвращаются)")

class TTSRequest(BaseModel):
    text: str = Field(..., examples=["Привет! Это проверочный синтез речи."])
    language: Optional[str] = Field("ru", examples=["ru"])
    speaker_wav: Optional[str] = Field(None, description="URL/путь до эталонного голоса (если поддерживается)")

# ======================
# LLM (Ollama)  /chat
# ======================

@app.post(
    "/chat",
    summary="Диалог с LLM (Ollama /api/chat)",
    description="Передай массив сообщений (system/user/assistant). Возвращает ответ Ollama в форме chat.",
    tags=["LLM"],
    response_model=ChatGatewayResponse,
)
async def chat(payload: ChatRequest = Body(...)):
    body = {
        "model": payload.model or "qwen2.5:3b-instruct-q4_K_M",
        "messages": [m.model_dump() for m in payload.messages],
        "stream": bool(payload.stream),
        "options": (payload.options.model_dump() if payload.options else ChatOptions().model_dump()),
    }
    r = await client.post(f"{OLLAMA}/api/chat", json=body)
    if r.is_error:
        raise HTTPException(r.status_code, r.text)
    return JSONResponse(r.json())

# ======================
# LLM (Ollama)  /message
# ======================

@app.post(
    "/message",
    summary="Один запрос (prompt) к LLM (Ollama /api/generate)",
    description="Удобно для простых одношаговых запросов. Под капотом вызывает /api/generate.",
    tags=["LLM"],
    response_model=GenerateGatewayResponse,
)
async def message(payload: MessageRequest = Body(...)):
    prompt = payload.prompt
    if not prompt:
        raise HTTPException(400, "Field 'prompt' is required")

    model = payload.model or "qwen2.5:3b-instruct-q4_K_M"
    system = payload.system or "Ты — русскоязычный ассистент. Всегда отвечай по-русски, кратко и грамотно."
    options = (payload.options.model_dump() if payload.options else MessageOptions().model_dump())
    stream = bool(payload.stream)

    req = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "options": options,
        "stream": stream,
    }

    # non-stream: простой возврат JSON
    if not stream:
        r = await client.post(f"{OLLAMA}/api/generate", json=req)
        if r.is_error:
            raise HTTPException(r.status_code, r.text)

        # Иногда Ollama может вернуть несколько JSON строк (редко) — склеим безопасно
        txt = r.text.strip()
        try:
            data = json.loads(txt)
            return JSONResponse(data)
        except json.JSONDecodeError:
            response_text = ""
            done_obj: Dict[str, Any] = {}
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
                    continue
            if not response_text and not done_obj:
                return JSONResponse({"raw": txt})
            merged = {"model": model, "response": response_text, "done": True}
            if done_obj:
                merged.update({k: v for k, v in done_obj.items() if k not in merged})
            return JSONResponse(merged)

    # stream=true: агрегируем строки сервера и возвращаем итог
    async with client.stream("POST", f"{OLLAMA}/api/generate", json=req) as resp:
        if resp.is_error:
            text = await resp.aread()
            raise HTTPException(resp.status_code, text.decode("utf-8", errors="ignore"))
        response_text = ""
        done_obj: Dict[str, Any] = {}
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

# ======================
# STT (Whisper)  /stt
# ======================

@app.post(
    "/stt",
    summary="Распознавание речи (Faster-Whisper)",
    description="Загрузи аудиофайл (multipart/form-data). Опционально укажи язык (например, ru).",
    tags=["STT"],
    response_model=STTResponse,
)
async def stt(file: UploadFile = File(..., description="Аудиофайл (wav/mp3/etc.)"),
              language: Optional[str] = Form(None, description="Подсказка языка, напр. 'ru'")):
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

# ======================
# TTS (XTTS)  /tts
# ======================

@app.post(
    "/tts",
    summary="Синтез речи (XTTS v2)",
    description="Принимает JSON с текстом и опциональными параметрами. Возвращает аудио (audio/wav).",
    tags=["TTS"],
)
async def tts(body: TTSRequest = Body(...)):
    if not body.text:
        raise HTTPException(400, "Field 'text' is required")

    r = await client.post(f"{XTTS}/tts", json=body.model_dump())
    if r.is_error:
        raise HTTPException(r.status_code, r.text)

    media = r.headers.get("content-type", "audio/wav")
    return Response(content=r.content, media_type=media, headers={
        "Content-Disposition": 'inline; filename="speech.wav"'
    })

# ======================
# IMAGE (ComfyUI)  /image/raw
# ======================

class ComfyPayload(BaseModel):
    # Оставляем гибкость — ComfyUI схемы зависят от workflow
    model_config = ConfigDict(extra="allow", json_schema_extra={
        "examples": [
            {
                "prompt": {
                    "3": {"inputs": {"text": "Astronaut riding a horse", "clip": ["5", 0]}, "class_type": "CLIPTextEncode"},
                    # ...
                }
            }
        ]
    })

@app.post(
    "/image/raw",
    summary="Прямой прокси к ComfyUI /prompt",
    description="Передай готовый JSON для ComfyUI. Ответ возвращается как есть.",
    tags=["Image"],
)
async def comfy_raw(payload: ComfyPayload = Body(...)):
    r = await client.post(f"{COMFY}/prompt", json=payload.model_dump())
    if r.is_error:
        raise HTTPException(r.status_code, r.text)
    return JSONResponse(r.json())
