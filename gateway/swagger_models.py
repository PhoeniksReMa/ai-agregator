from typing import Optional, List, Literal, Dict, Any

from pydantic import BaseModel, Field, ConfigDict



DEFAULT_MODEL  = "qwen2.5:3b-instruct-q5_K_M"

Role = Literal["system", "user", "assistant", "developer"]


class ChatMessage(BaseModel):
    role: Role = Field(..., json_schema_extra={"examples": ["user"]})
    content: str = Field(..., json_schema_extra={"examples": ["Привет! Кто ты?"]})


class ChatOptions(BaseModel):
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="Креативность")
    num_ctx: Optional[int] = Field(4096, ge=256, description="Контекст (токены)")
    num_predict: Optional[int] = Field(256, ge=256, description="Длинна вывода")
    top_p: Optional[float] = Field(None, ge=0, le=1)
    top_k: Optional[int] = Field(None, ge=0)
    repeat_penalty: Optional[float] = Field(None, ge=0)


class ChatRequest(BaseModel):
    model: Optional[str] = Field(DEFAULT_MODEL, description="Ollama model tag")
    messages: List[ChatMessage] = Field(..., description="История диалога")
    stream: Optional[bool] = Field(False, description="Стриминговый ответ")
    options: ChatOptions = ChatOptions()

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model": DEFAULT_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Ты — русскоязычный ассистент. Отвечай кратко.",
                        },
                        {
                            "role": "user",
                            "content": "Сформулируй правило трёх в одном предложении.",
                        },
                    ],
                    "stream": False,
                    "options": {"temperature": 0.5, "num_ctx": 2048, "num_predict": 256},
                }
            ]
        }
    )


class ChatGatewayResponse(BaseModel):
    model: Optional[str] = None
    done: Optional[bool] = None
    message: Optional[ChatMessage] = None

    model_config = ConfigDict(extra="allow")


class MessageOptions(BaseModel):
    temperature: Optional[float] = Field(0.2, ge=0, le=2)
    top_p: Optional[float] = Field(0.9, ge=0, le=1)
    repeat_penalty: Optional[float] = Field(1.1, ge=0)


class MessageRequest(BaseModel):
    prompt: str = Field(..., description="Текст запроса")
    model: Optional[str] = Field(DEFAULT_MODEL)
    system: Optional[str] = Field(
        "Ты — русскоязычный ассистент. Всегда отвечай по-русски, кратко и грамотно."
    )
    # Важно: без Optional и без default_factory
    options: MessageOptions = MessageOptions()
    stream: Optional[bool] = Field(False)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "prompt": "Суммируй: 1) Сборка; 2) Тест; 3) Деплой — в одном абзаце.",
                    "model": DEFAULT_MODEL,
                    "system": "Ты — русскоязычный ассистент. Отвечай кратко.",
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                    },
                    "stream": False,
                }
            ]
        }
    )


class GenerateGatewayResponse(BaseModel):
    model: Optional[str] = None
    response: Optional[str] = None
    done: Optional[bool] = None

    model_config = ConfigDict(extra="allow")


class STTSegment(BaseModel):
    start: float
    end: float
    text: str


class STTResponse(BaseModel):
    language: Optional[str] = Field(None, description="Детектированный язык (если есть)")
    text: str = Field(..., description="Итоговая транскрипция")
    segments: Optional[List[STTSegment]] = Field(
        None, description="Сегменты (если возвращаются)"
    )


class TTSRequest(BaseModel):
    text: str = Field(
        ..., json_schema_extra={"examples": ["Привет! Это проверочный синтез речи."]}
    )
    language: Optional[str] = Field("ru", json_schema_extra={"examples": ["ru"]})
    speaker_wav: Optional[str] = Field(
        None, description="URL/путь до эталонного голоса (если поддерживается)"
    )
class ComfyNode(BaseModel):
    class_type: str = Field(..., description="Имя класса узла ComfyUI, напр. SaveImage")
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Входные параметры узла. Значения могут быть примитивами или ссылками вида ['<node_id>', <out_idx>]"
    )

    model_config = ConfigDict(extra="allow")

class ComfyPayload(BaseModel):
    client_id: Optional[str] = Field(None, description="Произвольный идентификатор клиента/сессии")
    # ВАЖНО: теперь prompt — словарь {<node_id>: ComfyNode}
    prompt: Dict[str, ComfyNode] = Field(..., description="Граф ComfyUI")
    extra_data: Optional[Dict[str, Any]] = Field(
        None, description="Доп. метаданные (например extra_pnginfo/workflow)"
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "client_id": "demo-client",
                    "prompt": {
                        "1": {
                            "class_type": "EmptyLatentImage",
                            "inputs": {"width": 512, "height": 512, "batch_size": 1}
                        },
                        "2": {
                            "class_type": "CLIPTextEncode",
                            "inputs": {"text": "Astronaut riding a horse", "clip": ["5", 0]}
                        },
                        "3": {
                            "class_type": "KSampler",
                            "inputs": {
                                "seed": 123456789,
                                "steps": 20,
                                "cfg": 7,
                                "sampler_name": "euler",
                                "scheduler": "normal",
                                "denoise": 1.0,
                                "model": ["4", 0],
                                "positive": ["2", 0],
                                "negative": ["6", 0],
                                "latent_image": ["1", 0]
                            }
                        },
                        "4": {
                            "class_type": "CheckpointLoaderSimple",
                            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}
                        },
                        "5": {
                            "class_type": "CLIPSetLastLayer",
                            "inputs": {"stop_at_clip_layer": -2, "clip": ["4", 1]}
                        },
                        "6": {
                            "class_type": "CLIPTextEncode",
                            "inputs": {"text": "", "clip": ["5", 0]}
                        },
                        "7": {
                            "class_type": "VAEDecode",
                            "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
                        },
                        "8": {
                            "class_type": "SaveImage",
                            "inputs": {"images": ["7", 0], "filename_prefix": "comfy_out"}
                        }
                    },
                    "extra_data": {
                        "extra_pnginfo": {"workflow": {}}
                    }
                }
            ]
        },
    )


class OpenAIMessage(BaseModel):
    role: Role = Field(..., examples=["user"])
    content: str = Field(..., examples=["Hello! Who are you?"])

class OpenAIChatRequest(BaseModel):
    model: str = Field(..., description="Модель OpenAI (например 'gpt-4o-mini')")
    messages: List[OpenAIMessage] = Field(..., description="История диалога")
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="Креативность")
    top_p: Optional[float] = Field(1.0, ge=0, le=1)
    max_tokens: Optional[int] = Field(None, description="Ограничение длины вывода")
    stream: Optional[bool] = Field(False, description="Стриминговый ответ")
    extra: Optional[Dict[str, Any]] = Field(None, description="Дополнительные параметры OpenAI")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "Ты — умный ассистент, отвечай по-русски."},
                        {"role": "user", "content": "Объясни правило трёх в одном предложении."}
                    ],
                    "temperature": 0.7,
                    "stream": False
                }
            ]
        }
    )