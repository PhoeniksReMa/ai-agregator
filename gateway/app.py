import os, base64, httpx
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
XTTS   = os.getenv("XTTS_URL",   "http://localhost:8021")
WHISPER= os.getenv("WHISPER_URL","http://localhost:8022")
COMFY  = os.getenv("COMFY_URL",  "http://localhost:8188")

app = FastAPI(title="Local AI Aggregator")

# ----- Chat (LLM: Mistral via Ollama) -----
class ChatIn(BaseModel):
    prompt: str
    system: str | None = "You are a helpful assistant."
    model: str | None = "mistral:7b-instruct-q4_K_M"

@app.post("/chat")
async def chat(inp: ChatIn):
    payload = {
        "model": inp.model,
        "prompt": f"<s>[INST] {inp.system}\n{inp.prompt} [/INST]"
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA}/api/generate", json=payload)
        r.raise_for_status()
        # Ollama stream=true по умолчанию; для простоты — ждём full text
        data = r.json()
    return {"text": data.get("response", "")}

# ----- TTS -----
class TTSIn(BaseModel):
    text: str
    language: str = "ru"

@app.post("/tts")
async def tts(inp: TTSIn):
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{XTTS}/tts", json=inp.model_dump())
        r.raise_for_status()
        # Проксируем файл как base64 (или сделайте /tts/file для стрима)
        audio_b = r.content
    b64 = base64.b64encode(audio_b).decode()
    return {"audio_wav_base64": b64}

# ----- STT -----
@app.post("/stt")
async def stt(file: UploadFile = File(...), language: str = Form(None)):
    async with httpx.AsyncClient(timeout=300) as client:
        files = {"file": (file.filename, await file.read(), file.content_type or "audio/wav")}
        data  = {"language": language} if language else {}
        r = await client.post(f"{WHISPER}/transcribe", files=files, data=data)
        r.raise_for_status()
        return r.json()

# ----- SD (ComfyUI): простая прокси-дефолт -----
class ImgIn(BaseModel):
    prompt: str

@app.post("/image")
async def image(inp: ImgIn):
    # Здесь можно собрать JSON-граф ComfyUI и отправить на /prompt
    return {"status":"todo", "hint":"Соберите граф под ваш чекпоинт и дерните ComfyUI /prompt"}
