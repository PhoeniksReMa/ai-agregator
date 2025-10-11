from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import tempfile, os, logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("whisper")

app = FastAPI()
_model = None
_model_name = os.getenv("WHISPER_MODEL", "small")           # было "medium"
_compute = os.getenv("WHISPER_COMPUTE", "int8_float16")     # более щадящий VRAM

def get_model():
    global _model
    if _model is not None:
        return _model
    # сначала пробуем CUDA, при ошибке — CPU
    try:
        log.info(f"Loading Whisper model={_model_name} device=cuda compute={_compute}")
        _model = WhisperModel(_model_name, device="cuda", compute_type=_compute)
    except Exception as e:
        log.warning(f"CUDA load failed: {e}. Falling back to CPU (compute=int8).")
        _model = WhisperModel(_model_name, device="cpu", compute_type="int8")
    return _model

@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "ok", "model": _model_name, "compute": _compute}
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form(None)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        segments, info = get_model().transcribe(path, language=language)
        text = "".join(s.text for s in segments)
        return {"text": text, "language": getattr(info, "language", language)}
    finally:
        try: os.remove(path)
        except: pass
