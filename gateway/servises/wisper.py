import asyncio

import httpx
from fastapi import UploadFile, File, Form, HTTPException, APIRouter
from fastapi.responses import JSONResponse

from gateway.settings import client, WHISPER
from gateway.swagger_models import STTResponse, Optional

router = APIRouter(tags=["WHISPER"])

@router.post(
    "/stt",
    summary="Распознавание речи (Faster-Whisper)",
    description="Загрузи аудиофайл (multipart/form-data). Опционально укажи язык (например, ru).",
    tags=["STT"],
    response_model=STTResponse,
)
async def stt(file: UploadFile = File(..., description="Аудиофайл (wav/mp3/etc.)"),
              language: Optional[str] = Form(None, description="Подсказка языка, напр. 'ru'")):
    files = {"file": (file.filename, await file.read(), file.content_type or "application/octet-stream")}
    data = {}
    if language:
        data["language"] = language

    for attempt in range(2):
        try:
            r = await client.post(f"{WHISPER}/transcribe", files=files, data=data)
            if r.is_error:
                raise HTTPException(r.status_code, r.text)
            return JSONResponse(r.json())
        except httpx.RemoteProtocolError as e:
            if attempt == 0:
                await asyncio.sleep(1.0)
                continue
            raise HTTPException(status_code=502, detail=f"whisper upstream disconnected: {e}")
