FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

# базовые пакеты
RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# обновить pip и поставить PyTorch с CUDA 12.1
RUN python3 -m pip install --upgrade pip
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# скачать ComfyUI
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI
WORKDIR /opt/ComfyUI

# зависимости ComfyUI
RUN pip3 install --no-cache-dir -r requirements.txt

# порт и запуск
ENV CLI_ARGS="--listen 0.0.0.0 --port 8188"
EXPOSE 8188
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"]