from fastapi import UploadFile, File, Form, HTTPException, APIRouter
from fastapi.responses import JSONResponse

from ..settings import WHISPER
from ..dependencies import HttpDep
from ..swagger_models import STTResponse, Optional

router = APIRouter(tags=["WHISPER"])

@router.post(
    "/stt",
    summary="Распознавание речи (Faster-Whisper)",
    description="Загрузи аудиофайл (multipart/form-data). Опционально укажи язык (например, ru).",
    tags=["STT"],
    response_model=STTResponse,
)
async def stt(http: HttpDep, file: UploadFile = File(...), language: Optional[str] = Form(None)):
    files = {"file": (file.filename, await file.read(), file.content_type or "application/octet-stream")}
    data = {"language": language} if language else {}
    r = await http.post(f"{WHISPER}/transcribe", files=files, data=data)
    if r.is_error:
        raise HTTPException(r.status_code, r.text)
    return JSONResponse(r.json())