from typing import Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

# Модель по умолчанию для Ollama
model = "qwen2.5:3b-instruct-q4_K_M"

# Роли сообщений для чата
Role = Literal["system", "user", "assistant"]


# ---------- Chat (Ollama /api/chat, /api/generate) ----------

class ChatMessage(BaseModel):
    role: Role = Field(...)
    content: str = Field(...)

    # Пример для этой модели (не в Field, чтобы не ломать OpenAPI)
    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {"role": "user", "content": "Привет! Кто ты?"}
        ]
    })


class ChatOptions(BaseModel):
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="Креативность")
    num_ctx: Optional[int] = Field(4096, ge=256, description="Контекст (токены)")
    num_predict: Optional[int] = Field(256, ge=1, description="Длина вывода")
    top_p: Optional[float] = Field(None, ge=0, le=1)
    top_k: Optional[int] = Field(None, ge=1)
    repeat_penalty: Optional[float] = Field(None, ge=0)


class ChatRequest(BaseModel):
    model: Optional[str] = Field(model, description="Ollama model tag")
    messages: List[ChatMessage] = Field(..., description="История диалога")
    stream: bool = Field(False, description="Стриминговый ответ")
    options: ChatOptions = Field(default_factory=ChatOptions, description="Параметры генерации")

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Ты — русскоязычный ассистент. Отвечай кратко."},
                    {"role": "user", "content": "Сформулируй правило трёх в одном предложении."}
                ],
                "stream": False,
                "options": {"temperature": 0.5, "num_ctx": 2048, "num_predict": 256}
            }
        ]
    })


class ChatGatewayResponse(BaseModel):
    model: Optional[str] = None
    done: Optional[bool] = None
    message: Optional[ChatMessage] = None

    # позволяем дополнительные поля от Ollama
    model_config = ConfigDict(extra="allow")


class MessageOptions(BaseModel):
    temperature: Optional[float] = Field(0.2, ge=0, le=2)
    top_p: Optional[float] = Field(0.9, ge=0, le=1)
    repeat_penalty: Optional[float] = Field(1.1, ge=0)


class MessageRequest(BaseModel):
    prompt: str = Field(..., description="Текст запроса")
    model: Optional[str] = Field(model)
    system: Optional[str] = Field("Ты — русскоязычный ассистент. Всегда отвечай по-русски, кратко и грамотно.")
    options: MessageOptions = Field(default_factory=MessageOptions)
    stream: bool = Field(False)

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "prompt": "Суммируй: 1) Сборка; 2) Тест; 3) Деплой — в одном абзаце.",
                "model": model,
                "system": "Ты — русскоязычный ассистент. Отвечай кратко.",
                "options": {"temperature": 0.3, "top_p": 0.9, "repeat_penalty": 1.1},
                "stream": False
            }
        ]
    })


class GenerateGatewayResponse(BaseModel):
    model: Optional[str] = None
    response: Optional[str] = None
    done: Optional[bool] = None

    model_config = ConfigDict(extra="allow")


# ---------- STT (Whisper) ----------

class STTSegment(BaseModel):
    start: float
    end: float
    text: str


class STTResponse(BaseModel):
    language: Optional[str] = Field(None, description="Детектированный язык (если есть)")
    text: str = Field(..., description="Итоговая транскрипция")
    segments: Optional[List[STTSegment]] = Field(None, description="Сегменты (если возвращаются)")


# ---------- TTS (XTTS) ----------

class TTSRequest(BaseModel):
    text: str = Field(...)
    language: Optional[str] = Field("ru")
    speaker_wav: Optional[str] = Field(None, description="URL/путь до эталонного голоса (если поддерживается)")

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {"text": "Привет! Это проверочный синтез речи.", "language": "ru"}
        ]
    })


# ---------- ComfyUI raw payload ----------

class ComfyPayload(BaseModel):
    # Явное поле делает схему понятнее; разрешаем любые доп. ключи
    prompt: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow", json_schema_extra={
        "examples": [
            {
                "prompt": {
                    "3": {
                        "inputs": {"text": "Astronaut riding a horse", "clip": ["5", 0]},
                        "class_type": "CLIPTextEncode"
                    }
                }
            }
        ]
    })
