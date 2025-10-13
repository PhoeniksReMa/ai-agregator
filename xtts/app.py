from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from TTS.api import TTS
import tempfile, os
import torch

app = FastAPI()
_model = None

def get_model():
    global _model
    if _model is None:
        use_gpu = torch.cuda.is_available()
        _model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)
    return _model

class TTSIn(BaseModel):
    text: str
    speaker_wav: str | None = None
    language: str = "ru"

@app.get("/health")
def health():
    try:
        m = get_model()
        return {"status": "ok", "cuda": torch.cuda.is_available(), "cuda_version": torch.version.cuda}
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

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
