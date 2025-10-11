FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip
RUN pip3 install faster-whisper==1.0.1 fastapi uvicorn[standard] python-multipart

WORKDIR /app
COPY app.py /app/app.py
EXPOSE 8022
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8022"]
