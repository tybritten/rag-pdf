from __future__ import annotations

import asyncio
import os
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
# from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from starlette.responses import StreamingResponse

from ai import AIPipeline
from models import (ConfigGetResponse, GeneratePostRequest,
                    GeneratePostResponse, SourcesGetResponse)

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
DEFAULT_CHAT_MODEL = os.environ.get("CHAT_MODEL", None)
CHAT_MODEL_BASE_URL = os.environ.get("CHAT_MODEL_BASE_URL", None)
TOP_K = os.environ.get("TOP_K", 5)
CUTOFF = os.environ.get("CUTOFF", 0.7)
STREAMING = os.environ.get("STREAMING", True)
TEMP = os.environ.get("TEMP", 0.2)
MAX_TOKENS = os.environ.get("MAX_TOKENS", 512)
DB_PATH = os.environ.get("DB_PATH", "./db")
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "If you don't know the answer to a question, please don't share false information.",
)

ai = AIPipeline(
    chat_model=DEFAULT_CHAT_MODEL,
    chat_model_url=CHAT_MODEL_BASE_URL,
    embed_model=EMBED_MODEL,
    similarity_cutoff=CUTOFF,
    top_k=TOP_K,
    max_tokens=MAX_TOKENS,
    temp=TEMP,
    system_prompt=SYSTEM_PROMPT,
    db_path=DB_PATH,
    streaming=STREAMING,
)

app = FastAPI(
    title="Retrieval Augmented Generation API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

index, total_collection = ai.load_data()
llm = ai.load_llm()


@app.get("/config", response_model=ConfigGetResponse)
def get_config() -> ConfigGetResponse:
    config = ai.get_config()
    models = list(ai.get_models().keys())
    logger.info(f"config: {config}")
    logger.info(f"models: {models}")
    return ConfigGetResponse(defaultConfig=config, models=models)


@app.post("/generate", response_model=GeneratePostResponse)
def post_generate(body: GeneratePostRequest) -> GeneratePostResponse:
    if body.tags:
        filters = ai.create_filters(body.tags)
    else:
        filters = None
    system_prompt = None
    if body.config:
        config = body.config
        cutoff = config["similarityCutoff"] if "similarityCutoff" in config else None
        cutoff = cutoff / 100 if cutoff > 1.0 else cutoff
        top_k = config["topK"] if "topK" in config else None
        temp = config["modelTemperature"] if "modelTemperature" in config else None
        max_tokens = config["maxOutputTokens"] if "maxOutputTokens" in config else None
        system_prompt = config["systemPrompt"] if "systemPrompt" in config else None
    else:
        cutoff = None
        top_k = None
        temp = None
        max_tokens = None
    query_engine = ai.create_query_engine(
        index=index, filters=filters, cutoff=cutoff, top_k=top_k
    )

    model = body.model
    model_options = {
        "cutoff": cutoff,
        "top_k": top_k,
        "model": model,
        "tags": body.tags,
        "model_temperature": temp,
        "max_output_tokens": max_tokens,
    }
    (
        response,
        output_nodes,
    ) = ai.generate_response(
        body.query, system_prompt, model, model_options, query_engine
    )
    return StreamingResponse(
        ai.output_stream(response, output_nodes), media_type="application/x-ndjson"
    )


@app.get("/sources", response_model=List[SourcesGetResponse])
def get_sources() -> List[SourcesGetResponse]:
    sources = ai.source_list(total_collection)
    return [SourcesGetResponse(tag=tag, items=sources[tag]) for tag in sources]


if __name__ == "__main__":
    uvicorn.run(
        "main:app", host="0.0.0.0", port=5001, log_level="debug", reload=True, workers=1
    )
