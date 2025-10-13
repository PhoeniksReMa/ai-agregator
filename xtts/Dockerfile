FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg git && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

RUN python3 -m pip install --no-cache-dir TTS==0.22.0 fastapi uvicorn[standard] soundfile

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt || true

COPY app.py /app/app.py
EXPOSE 8021
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8021"]
