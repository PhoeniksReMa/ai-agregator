from fastapi import Body, HTTPException, APIRouter

from gateway.settings import COMFY
from gateway.dependencies import HttpDep
from gateway.swagger_models import ComfyPayload

router = APIRouter(tags=["COMFY"])


@router.post(
    "/image/raw",
    summary="Прямой прокси к ComfyUI /prompt",
    description="Передай готовый JSON для ComfyUI. Ответ возвращается как JSON ComfyUI.",
)
async def comfy_raw(http: HttpDep, payload: ComfyPayload = Body(...)):
    r = await http.post(f"{COMFY}/prompt", json=payload.model_dump(exclude_none=True))
    if r.is_error:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()
