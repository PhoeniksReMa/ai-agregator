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
    options: MessageOptions = MessageOptions()
    stream: Optional[bool] = Field(False)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "prompt": "Суммируй: 1) Сборка; 2) Тест; 3) Деплой — в одном абзаце.",
                    "model": DEFAULT_MODEL,
                    "system": "Ты — русскоязычный ассистент. Отвечай кратко.",
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
    class_type: str
    inputs: Dict[str, Any]

# Полный полезный payload для ComfyUI (/prompt)
class ComfyPayload(BaseModel):
    client_id: Optional[str] = Field(None, description="ID клиента/сессии")
    prompt: Dict[str, ComfyNode] = Field(..., description="Граф ComfyUI")
    extra_data: Optional[Dict[str, Any]] = Field(
        None, description="Доп. метаданные (extra_pnginfo/workflow и т.п.)"
    )

    model_config = ConfigDict(extra="allow")

# Простой внешний запрос (то, что шлёт пользователь)
class Text2ImageRequest(BaseModel):
    prompt: str = Field(..., description="Позитивный текстовый промпт")
    negative: Optional[str] = Field("", description="Негативный промпт")
    width: int = Field(768, ge=64, multiple_of=8, description="Ширина")
    height: int = Field(768, ge=64, multiple_of=8, description="Высота")
    batch_size: int = Field(1, ge=1, le=8, description="Размер батча")
    steps: int = Field(20, ge=1, le=100, description="Шаги диффузии")
    cfg: float = Field(7.0, ge=0, le=30, description="Guidance scale")
    sampler_name: str = Field("dpmpp_2m", description="Сэмплер (euler, dpmpp_2m, ...)")
    scheduler: str = Field("karras", description="Планировщик (normal, karras, ...)")
    denoise: float = Field(1.0, ge=0.0, le=1.0, description="Сила денойза")
    seed: int = Field(-1, description="Сид (-1 = случайный на стороне Comfy)")
    # Технические настройки (оставляем по умолчанию, но можно переопределить)
    ckpt_name: str = Field("sd_xl_base_1.0.safetensors", description="Чекпоинт")
    stop_at_clip_layer: int = Field(-2, description="CLIP слой для SDXL")
    filename_prefix: str = Field("comfy_out", description="Префикс сохранения")
    client_id: Optional[str] = Field(None, description="ID клиента/сессии")
    extra_data: Optional[Dict[str, Any]] = Field(
        None, description="Метаданные, попадут в extra_pnginfo"
    )

def build_comfy_payload(req: Text2ImageRequest) -> ComfyPayload:
    """
    Собирает базовый SDXL txt2img граф под ComfyUI из простого запроса пользователя.
    """
    nodes: Dict[str, ComfyNode] = {
        # 4 — грузим SDXL Base: даёт (0) UNet, (1) CLIP, (2) VAE
        "4": ComfyNode(
            class_type="CheckpointLoaderSimple",
            inputs={"ckpt_name": req.ckpt_name},
        ),
        # 5 — настраиваем CLIP слой (для SDXL обычно -2)
        "5": ComfyNode(
            class_type="CLIPSetLastLayer",
            inputs={"clip": ["4", 1], "stop_at_clip_layer": req.stop_at_clip_layer},
        ),
        # 2 — позитивный промпт
        "2": ComfyNode(
            class_type="CLIPTextEncode",
            inputs={"clip": ["5", 0], "text": req.prompt},
        ),
        # 6 — негативный промпт
        "6": ComfyNode(
            class_type="CLIPTextEncode",
            inputs={"clip": ["5", 0], "text": req.negative or ""},
        ),
        # 1 — пустой латент нужного размера
        "1": ComfyNode(
            class_type="EmptyLatentImage",
            inputs={
                "batch_size": req.batch_size,
                "height": req.height,
                "width": req.width,
            },
        ),
        # 3 — диффузия
        "3": ComfyNode(
            class_type="KSampler",
            inputs={
                "seed": req.seed,
                "steps": req.steps,
                "cfg": req.cfg,
                "sampler_name": req.sampler_name,
                "scheduler": req.scheduler,
                "denoise": req.denoise,
                "model": ["4", 0],       # UNet
                "positive": ["2", 0],    # + prompt
                "negative": ["6", 0],    # - prompt
                "latent_image": ["1", 0] # стартовый латент
            },
        ),
        # 7 — декодим в пиксели
        "7": ComfyNode(
            class_type="VAEDecode",
            inputs={"samples": ["3", 0], "vae": ["4", 2]},
        ),
        # 8 — сохраняем (если вы сохраняете на стороне Comfy)
        "8": ComfyNode(
            class_type="SaveImage",
            inputs={"images": ["7", 0], "filename_prefix": req.filename_prefix},
        ),
    }

    extra = req.extra_data or {}
    # Запишем исходный запрос в метаданные — удобно для отладки
    extra_pnginfo = extra.get("extra_pnginfo", {})
    extra_pnginfo.setdefault("workflow", {})
    extra_pnginfo["request"] = req.model_dump()
    extra["extra_pnginfo"] = extra_pnginfo

    return ComfyPayload(
        client_id=req.client_id,
        prompt=nodes,
        extra_data=extra,
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
    extra: Optional[Dict[str, Any]] = Field(None, description="Дополнительные параметры OpenAI")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "Ты — умный ассистент, отвечай по-русски."},
                        {"role": "user", "content": "Объясни правило трёх в одном предложении."}
                    ]
                }
            ]
        }
    )