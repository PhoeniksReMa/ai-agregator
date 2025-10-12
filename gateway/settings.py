import os

OLLAMA = os.getenv("OLLAMA_URL", "http://ollama:11434")
XTTS   = os.getenv("XTTS_URL",   "http://xtts:8021")
WHISPER= os.getenv("WHISPER_URL","http://whisper:8022")
COMFY  = os.getenv("COMFY_URL",  "http://comfyui:8188")