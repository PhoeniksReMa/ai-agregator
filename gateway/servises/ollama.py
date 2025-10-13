from typing import Any, Dict
import json

from fastapi import Body, HTTPException, APIRouter
from gateway.settings import OLLAMA
from gateway.dependencies import HttpDep
from gateway.swagger_models import (
    ChatGatewayResponse, ChatRequest, ChatOptions,
    GenerateGatewayResponse, MessageRequest, MessageOptions,
)

router = APIRouter(tags=["OLLAMA"])


@router.post(
    "/chat",
    summary="Диалог с LLM (Ollama /api/chat)",
    description="Передай массив сообщений (system/user/assistant). Возвращает ответ Ollama в форме chat.",
    tags=["OLLAMA"],
    response_model=ChatGatewayResponse,
)
async def chat(http: HttpDep, payload):
    body = {
        "model": payload.model or "qwen2.5:3b-instruct-q4_K_M",
        "messages": [m.model_dump(exclude_none=True) for m in payload.messages],
        "stream": bool(payload.stream),
        "options": (payload.options.model_dump(exclude_none=True)
                    if payload.options else ChatOptions().model_dump(exclude_none=True)),
    }
    r = await http.post(f"{OLLAMA}/api/chat", json=body)
    if r.is_error:
        raise HTTPException(r.status_code, r.text)
    # FastAPI сам вернёт JSON
    return r.json()


@router.post(
    "/message",
    summary="Один запрос (prompt) к LLM (Ollama /api/generate)",
    description="Удобно для простых одношаговых запросов. Под капотом вызывает /api/generate.",
    tags=["OLLAMA"],
    response_model=GenerateGatewayResponse,
)
async def message(http: HttpDep, payload):
    if not payload.prompt:
        raise HTTPException(400, "Field 'prompt' is required")

    model = payload.model or "qwen2.5:3b-instruct-q4_K_M"
    system = payload.system or "Ты — русскоязычный ассистент. Всегда отвечай по-русски, кратко и грамотно."
    options = (payload.options.model_dump(exclude_none=True)
               if payload.options else MessageOptions().model_dump(exclude_none=True))
    stream = bool(payload.stream)

    req = {"model": model, "prompt": payload.prompt, "system": system,
           "options": options,
           "stream": stream}

    # non-stream: обычный JSON-ответ
    if not stream:
        r = await http.post(f"{OLLAMA}/api/generate", json=req)
        if r.is_error:
            raise HTTPException(r.status_code, r.text)

        txt = r.text.strip()
        # 1) нормальный JSON
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            # 2) редкий случай: несколько JSON-строк подряд
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
                return {"raw": txt}
            merged = {"model": model, "response": response_text, "done": True}
            if done_obj:
                merged.update({k: v for k, v in done_obj.items() if k not in merged})
            return merged

    # stream=true: читаем построчно и собираем итог
    # (опц.) можно дать бесконечный таймаут на стрим: timeout=None
    async with http.stream("POST", f"{OLLAMA}/api/generate", json=req) as resp:
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
        return merged
