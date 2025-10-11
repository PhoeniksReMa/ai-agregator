from fastapi import FastAPI, UploadFile, File, Form
from faster_whisper import WhisperModel
import tempfile, os

app = FastAPI()
_model = WhisperModel("medium", device="cuda", compute_type="float16")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form(None)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    segments, info = _model.transcribe(tmp_path, language=language)
    text = "".join([s.text for s in segments])
    os.remove(tmp_path)
    return {"text": text, "language": info.language if hasattr(info, "language") else language}
