# gateway/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY app.py /app/app.py
# + python-multipart (нужен для форм и UploadFile)
RUN pip install --no-cache-dir fastapi uvicorn[standard] httpx pydantic python-multipart
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]