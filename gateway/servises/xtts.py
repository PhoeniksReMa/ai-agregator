from fastapi import Body, HTTPException, APIRouter
from fastapi.responses import Response

from gateway.settings import XTTS
from gateway.dependencies import HttpDep
from gateway.swagger_models import TTSRequest

router = APIRouter(tags=["TTS"])

@router.post("/tts", summary="Синтез речи (XTTS v2)")
async def tts(http: HttpDep, body: TTSRequest = Body(...)):
    r = await http.post(f"{XTTS}/tts", json=body.model_dump())
    if r.is_error:
        raise HTTPException(r.status_code, r.text)
    media = r.headers.get("content-type", "audio/wav")
    return Response(content=r.content, media_type=media, headers={
        "Content-Disposition": 'inline; filename="speech.wav"'
    })
