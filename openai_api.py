# coding=utf-8
# Implements API for ChatGLM3-6B in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.

# 请在当前目录运行

import time
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer






@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: str = None
    name: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    functions: Optional[Union[dict, List[dict]]] = None
    # Additional parameters
    repetition_penalty: Optional[float] = 1.1


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="chatglm3-6b")
    return ModelList(data=[model_card])

# 假设您有一个 embeddings_model 来获取嵌入
class EmbeddingsRequest(BaseModel):
    input: list 
    model: str
    encoding_format: str = "tokenized"

class EmbeddingsResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict


@app.post('/v1/embeddings', response_model=EmbeddingsResponse)
async def get_embeddings(request: EmbeddingsRequest):
    # 提取文本和模型ID
    text = request.input
    model_id = request.model
    
    embedding_data = []
    chunk_num = 100
    for i in range(0, len(text), 100):
        if len(text) > i + 100:
            chunk = text[i:i + 100]
        else:
            chunk = text[i:]
        embeddings = embeddings_model.encode(chunk)
        
        for r in embeddings:
            embedding_data.append(r.tolist())
    
    embedding_data = [
        {"embedding": r, 
        "index": i, 
        "object": "embedding"} for i, r in enumerate(embedding_data)]
    
    # 构建响应数据
    response_data = EmbeddingsResponse(
        data=embedding_data,
        model=model_id,
        object="list",
        usage={
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    )

    return response_data


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    embeddings_model = SentenceTransformer('moka-ai/m3e-base')
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
