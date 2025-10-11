FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg git && rm -rf /var/lib/apt/lists/*
# CPU BLAS может хватать, но при желании добавьте cuBLAS dev в base-образ
RUN pip3 install --upgrade pip

# Coqui TTS + сервер
RUN pip3 install TTS==0.22.0 fastapi uvicorn[standard] soundfile

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt || true
COPY app.py /app/app.py

EXPOSE 8021
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8021"]