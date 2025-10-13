from fastapi import APIRouter, Body, HTTPException

import httpx

from gateway.settings import COMFY, SERVER_URL
from gateway.swagger_models import build_simple_comfy_payload, SimpleTxtRequest

HTTP_TIMEOUT=httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)
router = APIRouter(tags=["Image Jobs"])

@router.post("/image/jobs", summary="Создать задачу генерации (txt only)", status_code=202)
async def create_image_job(body: SimpleTxtRequest = Body(...)):
    payload = build_simple_comfy_payload(body.text, body.client_id)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(f"{COMFY}/prompt", json=payload)
        if r.is_error:
            ct = r.headers.get("content-type", "")
            detail = r.json() if "application/json" in ct else r.text
            raise HTTPException(status_code=r.status_code, detail=detail)
        responce = r.json()
        return {"url": f'{SERVER_URL}/history/{responce["prompt_id"]}'}
