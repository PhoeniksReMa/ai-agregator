from fastapi import Body, HTTPException
from fastapi.responses import Response

from gateway.app import app, client, XTTS
from gateway.swagger_models import TTSRequest


@app.post(
    "/tts",
    summary="Синтез речи (XTTS v2)",
    description="Принимает JSON с текстом и опциональными параметрами. Возвращает аудио (audio/wav).",
    tags=["TTS"],
)
async def tts(body: TTSRequest = Body(...)):
    if not body.text:
        raise HTTPException(400, "Field 'text' is required")

    r = await client.post(f"{XTTS}/tts", json=body.model_dump())
    if r.is_error:
        raise HTTPException(r.status_code, r.text)

    media = r.headers.get("content-type", "audio/wav")
    return Response(content=r.content, media_type=media, headers={
        "Content-Disposition": 'inline; filename="speech.wav"'
    })
