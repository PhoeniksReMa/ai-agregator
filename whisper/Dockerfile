FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install --no-cache-dir --extra-index-url https://download.opennmt.net/pip/cu121 \
    ctranslate2==4.5.0 \
 && python3 -m pip install --no-cache-dir faster-whisper==1.0.3 fastapi uvicorn[standard] python-multipart

WORKDIR /app
COPY app.py /app/app.py
EXPOSE 8022
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8022"]
