from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI
import json
from fastapi.openapi.utils import get_openapi

from gateway.servises import xtts, wisper, ollama, comfy

import gateway.swagger_models as sm
print("swagger_models loaded from:", sm.__file__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0, read=120.0))
    try:
        yield
    finally:
        await app.state.http.aclose()

app = FastAPI(title="Local AI Gateway", lifespan=lifespan)

app.include_router(xtts.router)
app.include_router(wisper.router)
app.include_router(ollama.router)
app.include_router(comfy.router)

@app.get("/health")
async def health():
    return {"ok": True}
