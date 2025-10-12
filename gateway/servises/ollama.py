from typing import Any, Dict
import json

from fastapi import Body, HTTPException
from fastapi.responses import JSONResponse

from gateway.app import app, client, OLLAMA
from gateway.swagger_models import  ChatGatewayResponse, ChatRequest, ChatOptions, \
    GenerateGatewayResponse, MessageRequest, MessageOptions


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
