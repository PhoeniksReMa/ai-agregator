from fastapi import APIRouter

from gateway.settings import COMFY
from gateway.swagger_models import Text2ImageRequest, ComfyPayload, build_comfy_payload
from fastapi.responses import JSONResponse
import httpx

router = APIRouter(tags=["COMFY"])


@router.post(
    "/image/raw",
    summary="Прямой прокси к ComfyUI /prompt",
    description="Передай готовый JSON для ComfyUI. Ответ возвращается как JSON ComfyUI.",
)
async def txt2img(req: Text2ImageRequest):
    payload: ComfyPayload = build_comfy_payload(req)

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{COMFY}/prompt", json=payload.model_dump(mode="json"))
        r.raise_for_status()
        return JSONResponse(content=r.json())