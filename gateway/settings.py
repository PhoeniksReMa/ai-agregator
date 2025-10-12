import os
import httpx

OLLAMA = os.getenv("OLLAMA_URL", "http://ollama:11434")
XTTS   = os.getenv("XTTS_URL",   "http://xtts:8021")
WHISPER= os.getenv("WHISPER_URL","http://whisper:8022")
COMFY  = os.getenv("COMFY_URL",  "http://comfyui:8188")

TIMEOUT = httpx.Timeout(120.0, connect=10.0, read=120.0)
client = httpx.AsyncClient(timeout=TIMEOUT)