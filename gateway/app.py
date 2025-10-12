from fastapi import FastAPI
from settings import client
from servises import comfy, ollama, wisper, xtts

app = FastAPI(
    title="Local AI Aggregator Gateway",
    description="Шлюз к локальным сервисам: Ollama (LLM), Faster-Whisper (STT), XTTS (TTS), ComfyUI (image).",
)

app.include_router(xtts.router, prefix="", tags=["TTS"])
app.include_router(wisper.router, prefix="", tags=["STT"])
app.include_router(ollama.router, prefix="", tags=["LLM"])
app.include_router(comfy.router, prefix="", tags=["Image"])


@app.on_event("shutdown")
async def _shutdown():
    await client.aclose()

@app.get("/health", summary="Проверка здоровья", tags=["Service"])
async def health():
    return {"status": "ok"}
