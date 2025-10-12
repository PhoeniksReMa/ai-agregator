from fastapi import Body, HTTPException, APIRouter
from fastapi.responses import JSONResponse

from gateway.settings import client, COMFY
from gateway.swagger_models import ComfyPayload

router = APIRouter(tags=["COMFY"])


@router.post(
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
