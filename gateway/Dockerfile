FROM python:3.11-slim
WORKDIR /app
COPY app.py /app/app.py

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]