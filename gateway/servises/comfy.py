from fastapi import APIRouter, Body, HTTPException, Response, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import httpx

from gateway.settings import COMFY
from gateway.swagger_models import build_simple_comfy_payload

HTTP_TIMEOUT=httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)
router = APIRouter(tags=["Image Jobs"])

# ----- Модель запроса на создание -----
class CreateImageJobRequest(BaseModel):
    text: str = Field(..., description="Позитивный текстовый промпт")
    client_id: str | None = Field(None, description="ID клиента/сессии (опционально)")
    meta: dict | None = Field(None, description="Любые метаданные: попадут в extra_pnginfo")

# ----- Вспомогательная проверка статуса в Comfy -----
async def _check_history_for_image(client: httpx.AsyncClient, job_id: str) -> tuple[bool, dict | None]:
    """
    Возвращает (is_ready, image_meta|None).
    Если is_ready=False — image_meta=None.
    Если is_ready=True — image_meta={'filename','subfolder','type'} из /history.
    """
    hr = await client.get(f"{COMFY}/history/{job_id}")
    if hr.status_code == 404:
        # Comfy ещё не знает о таком job_id или он уже удалён историей
        return False, None
    if hr.is_error:
        raise HTTPException(status_code=hr.status_code, detail=hr.text)

    data = hr.json() or {}
    hist = (data.get("history") or {}).get(job_id) or {}
    outputs = hist.get("outputs") or {}
    for node_out in outputs.values():
        imgs = node_out.get("images") or []
        if imgs:
            info = imgs[0]
            return True, {
                "filename": info["filename"],
                "subfolder": info.get("subfolder", ""),
                "type": info.get("type", "output"),
            }
    return False, None

# ===== 1) СОЗДАНИЕ ЗАДАЧИ =====
@router.post("/image/jobs", summary="Создать задачу генерации (txt only)", status_code=202)
async def create_image_job(body: CreateImageJobRequest = Body(...)):
    payload = build_simple_comfy_payload(body.text, body.client_id, body.meta)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(f"{COMFY}/prompt", json=payload)
        if r.is_error:
            ct = r.headers.get("content-type", "")
            detail = r.json() if "application/json" in ct else r.text
            raise HTTPException(status_code=r.status_code, detail=detail)

    job_id = r.json().get("prompt_id")
    if not job_id:
        raise HTTPException(500, "ComfyUI did not return prompt_id")

    # Возвращаем 202 + ссылки для клиента
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "status_url": f"/image/jobs/{job_id}?json=1",
            "result_url": f"/image/jobs/{job_id}",  # отдаёт картинку, когда будет готова
        },
        headers={"Retry-After": "2"},  # подсказка клиенту по пуллингу
    )

# ===== 2) ПОЛУЧЕНИЕ ГОТОВОГО ИЗОБРАЖЕНИЯ (ИЛИ СТАТУСА 202) =====
@router.get("/image/jobs/{job_id}", summary="Получить готовое изображение")
async def get_image_result(
    job_id: str,
    json_status: int = Query(0, alias="json", description="1 = вернуть JSON-статус вместо картинки"),
):
    """
    Если изображение готово — вернёт байты изображения (200).
    Если ещё не готово — вернёт 202 с JSON-статусом и Retry-After.
    Если job неизвестен — 404.
    """
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        is_ready, meta = await _check_history_for_image(client, job_id)

        if not is_ready:
            # Можно попытаться понять — есть ли вообще такой job (Comfy иногда сразу создаёт history)
            # Если Comfy не знает такой history — формально 202 тоже норм: задача ещё в очереди/исполняется.
            if json_status:
                return JSONResponse(
                    status_code=202,
                    content={"job_id": job_id, "status": "running"},
                    headers={"Retry-After": "2"},
                )
            # по умолчанию даём JSON, чтобы фронт мог опросить дальше
            return JSONResponse(
                status_code=202,
                content={"job_id": job_id, "status": "running"},
                headers={"Retry-After": "2"},
            )

        # Готово — тянем файл через /view
        vr = await client.get(
            f"{COMFY}/view",
            params={
                "filename": meta["filename"],
                "subfolder": meta["subfolder"],
                "type": meta["type"],
            },
        )
        if vr.status_code == 404:
            # Редко, но файл может переместиться/очиститься — считаем, что ещё не готово/временная задержка
            return JSONResponse(
                status_code=202,
                content={"job_id": job_id, "status": "processing"},
                headers={"Retry-After": "2"},
            )
        if vr.is_error or not vr.content:
            raise HTTPException(status_code=vr.status_code, detail=vr.text[:500])

        # Выбираем media type по расширению (Comfy обычно png/webp)
        mt = "image/png"
        name = meta["filename"].lower()
        if name.endswith(".webp"):
            mt = "image/webp"
        elif name.endswith(".jpg") or name.endswith(".jpeg"):
            mt = "image/jpeg"

        headers = {"Content-Disposition": f'inline; filename="{meta["filename"]}"'}
        return StreamingResponse(iter([vr.content]), media_type=mt, headers=headers)
