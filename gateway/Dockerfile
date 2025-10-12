FROM python:3.11-slim
WORKDIR /app

COPY gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway /app/gateway

EXPOSE 8080
CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8080"]