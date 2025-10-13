from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
import httpx
from gateway.swagger_models import SimpleTxtRequest, build_simple_comfy_payload
from gateway.settings import COMFY

router = APIRouter(tags=["COMFY"])


@router.post("/image/simple", summary="SDXL txt2img: только текст")
async def txt2img_simple(body: SimpleTxtRequest = Body(...)):
    payload = build_simple_comfy_payload(body.text, body.client_id, body.meta)

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{COMFY}/prompt", json=payload)
        if r.is_error:

            ct = r.headers.get("content-type", "")
            detail = r.json() if "application/json" in ct else r.text
            raise HTTPException(status_code=r.status_code, detail=detail)

        prompt_id = r.json().get("prompt_id")
        if not prompt_id:
            raise HTTPException(500, "ComfyUI did not return prompt_id")

        import asyncio, time
        deadline = time.time() + 120
        result_image_bytes = None

        while time.time() < deadline:
            hr = await client.get(f"{COMFY}/history/{prompt_id}")
            if hr.status_code == 200:
                data = hr.json()
                # вытаскиваем первый image (под ваш формат)
                try:
                    # Comfy стандартно кладёт пути к файлам; если используете API image endpoint — адаптируйте
                    first = next(iter(data["history"][prompt_id]["outputs"].values()))
                    img_info = first["images"][0]
                    # получить файл через /view или /view?filename=...&subfolder=...&type=output
                    vr = await client.get(
                        f"{COMFY}/view",
                        params={
                            "filename": img_info["filename"],
                            "subfolder": img_info.get("subfolder", ""),
                            "type": img_info.get("type", "output"),
                        }
                    )
                    if vr.status_code == 200:
                        result_image_bytes = vr.content
                        break
                except Exception:
                    pass
            await asyncio.sleep(0.5)

        if not result_image_bytes:
            raise HTTPException(504, "Timed out waiting for image")

        return StreamingResponse(iter([result_image_bytes]), media_type="image/png")