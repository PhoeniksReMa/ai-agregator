#!/usr/bin/env bash
set -e

# Загружаем Qwen 2.5 и Mistral, если ещё не скачаны
if ! ollama list | grep -q "qwen2.5"; then
  echo "Pulling Qwen 2.5 model..."
  ollama pull qwen2.5:7b-instruct-q4_K_M
fi

if ! ollama list | grep -q "mistral"; then
  echo "Pulling Mistral model..."
  ollama pull mistral:7b-instruct-q4_K_M
fi

# Создаём русскоязычную модель на базе Qwen (опционально)
if ! ollama list | grep -q "qwen25-ru"; then
  echo "Creating russian-tuned Qwen..."
  cat <<EOF | ollama create qwen25-ru -f -
FROM qwen2.5:7b-instruct-q4_K_M
SYSTEM "Ты — русскоязычный ассистент. Отвечай только на русском языке, грамотно и естественно."
EOF
fi

# Запускаем сервер Ollama
exec /bin/ollama serve
