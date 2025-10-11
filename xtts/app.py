from fastapi import FastAPI
from pydantic import BaseModel
import tempfile, os
from TTS.api import TTS
from fastapi.responses import FileResponse

app = FastAPI()
_model = None

def get_model():
    global _model
    if _model is None:
        # скачает xtts_v2 при первом запуске
        _model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    return _model

class TTSIn(BaseModel):
    text: str
    speaker_wav: str | None = None   # путь к эталонному голосу (опционально)
    language: str = "ru"             # "en","ru",...

@app.post("/tts")
def tts(inp: TTSIn):
    tts = get_model()
    wav_path = tempfile.mktemp(suffix=".wav")
    tts.tts_to_file(
        text=inp.text,
        file_path=wav_path,
        speaker_wav=inp.speaker_wav,
        language=inp.language,
    )
    return FileResponse(wav_path, media_type="audio/wav", filename="speech.wav")
