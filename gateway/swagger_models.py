from typing import Optional, List, Literal, Dict, Any, Union
import random

from pydantic import BaseModel, Field, ConfigDict
from gateway.settings import (
    IMG_WIDTH, IMG_HEIGHT, IMG_BATCH, IMG_STEPS, IMG_CFG,
    IMG_SAMPLER, IMG_SCHED, IMG_DENOISE, IMG_CKPT, IMG_CLIP_LAYER,
    IMG_PREFIX, NEGATIVE_DEFAULT
)


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

class SimpleTxtRequest(BaseModel):
    text: str = Field(..., description="Позитивный текстовый промпт")
    client_id: Optional[str] = Field(None, description="ID клиента/сессии")


def build_simple_comfy_payload(text: str, client_id: str | None) -> Dict[str, Any]:

    nodes: Dict[str, Any] = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": IMG_CKPT}},
        "5": {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["4", 1], "stop_at_clip_layer": IMG_CLIP_LAYER}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": text}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": NEGATIVE_DEFAULT}},
        "1": {"class_type": "EmptyLatentImage", "inputs": {
            "batch_size": IMG_BATCH, "height": IMG_HEIGHT, "width": IMG_WIDTH
        }},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": random.randint(0, 2**31 - 1) ,
            "steps": IMG_STEPS,
            "cfg": IMG_CFG,
            "sampler_name": IMG_SAMPLER,
            "scheduler": IMG_SCHED,
            "denoise": IMG_DENOISE,
            "model": ["4", 0],
            "positive": ["2", 0],
            "negative": ["6", 0],
            "latent_image": ["1", 0],
        }},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": IMG_PREFIX}},
    }

    return {
        "client_id": client_id,
        "prompt": nodes,
    }


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


# class STTSegment(BaseModel):
#     start: float
#     end: float
#     text: str
#
#
# class STTResponse(BaseModel):
#     language: Optional[str] = Field(None, description="Детектированный язык (если есть)")
#     text: str = Field(..., description="Итоговая транскрипция")
#     segments: Optional[List[STTSegment]] = Field(
#         None, description="Сегменты (если возвращаются)"
#     )
#
#
# class TTSRequest(BaseModel):
#     text: str
#     speaker: Optional[str] = None
#     speaker_wav: Optional[Union[str, List[str]]] = None
#     language: Optional[str] = "ru"
#     out_path: Optional[str] = "/tmp/out.wav"