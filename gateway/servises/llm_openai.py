from fastapi import APIRouter
from fastapi.responses import JSONResponse

from openaiGPT.servises import OpenAiAPIServise
from gateway.swagger_models import OpenAIChatRequest

router = APIRouter(tags=["OPENAI"])

@router.post(
    "/chat_complete",
    summary="Диалог с LLM (OpenAI /chat/completions)",
    description="Передай массив сообщений (developer/user/assistant). Возвращает ответ OpenAI в форме chat.",
)
async def chat_complete(payload: OpenAIChatRequest):
    openai_api_servise = OpenAiAPIServise()
    response = openai_api_servise.chat_complete(
        model=payload.model,
        messages=[m.model_dump() for m in payload.messages],
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
        stream=payload.stream,
        **(payload.extra or {}),
    )
    return JSONResponse(content=response)