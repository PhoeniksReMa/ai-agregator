from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
import httpx
from gateway.swagger_models import SimpleTxtRequest, build_simple_comfy_payload
from gateway.settings import COMFY
import httpx, asyncio, time


router = APIRouter(tags=["COMFY"])

TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)
POLL_DEADLINE_SECONDS = 600  # до 10 минут под тяжёлые SDXL 1024/hires
POLL_INTERVAL_SECONDS = 0.6

async def wait_and_fetch_first_image(client: httpx.AsyncClient, comfy_base: str, prompt_id: str) -> bytes:
    deadline = time.time() + POLL_DEADLINE_SECONDS
    last_err = None

    while time.time() < deadline:
        hr = await client.get(f"{comfy_base}/history/{prompt_id}")
        if hr.status_code == 200:
            data = hr.json()
            # История у Comfy обычно: {"history": {prompt_id: {"outputs": {node_id: {...}}}}}
            hist = data.get("history", {}).get(prompt_id, {})
            outputs = hist.get("outputs") or {}
            # ищем первый output, где есть images
            for node_id, node_out in outputs.items():
                imgs = node_out.get("images") or []
                if imgs:
                    img_info = imgs[0]
                    # Comfy стандартные поля:
                    filename = img_info["filename"]
                    subfolder = img_info.get("subfolder", "")
                    img_type = img_info.get("type", "output")  # обычно "output"

                    vr = await client.get(
                        f"{comfy_base}/view",
                        params={"filename": filename, "subfolder": subfolder, "type": img_type},
                    )
                    if vr.status_code == 200 and vr.content:
                        return vr.content
                    else:
                        last_err = f"/view {vr.status_code}: {vr.text[:200]}"

        elif hr.status_code >= 400:
            last_err = f"/history {hr.status_code}: {hr.text[:200]}"

        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(last_err or "Timed out waiting for ComfyUI result")


@router.post("/image/simple", summary="SDXL txt2img: только текст")
async def txt2img_simple(body: SimpleTxtRequest = Body(...)):
    payload = build_simple_comfy_payload(body.text, body.client_id, body.meta)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(f"{COMFY}/prompt", json=payload)
        if r.is_error:
            ct = r.headers.get("content-type", "")
            detail = r.json() if "application/json" in ct else r.text
            raise HTTPException(status_code=r.status_code, detail=detail)

        prompt_id = r.json().get("prompt_id")
        if not prompt_id:
            raise HTTPException(500, "ComfyUI did not return prompt_id")

        try:
            img_bytes = await wait_and_fetch_first_image(client, COMFY, prompt_id)
        except TimeoutError as e:
            raise HTTPException(504, f"Timeout waiting image: {e}")

        # отдаём как PNG (Comfy по умолчанию png; если у тебя webp — поменяй тип)
        return StreamingResponse(iter([img_bytes]), media_type="image/png")