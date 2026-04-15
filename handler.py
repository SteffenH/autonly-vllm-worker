"""RunPod Serverless Handler for vLLM with OpenAI-compatible API."""

import os
import runpod
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import build_async_engine_client
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
import asyncio
import json

MODEL_NAME = os.getenv("MODEL_NAME", "monkeyslikebananas/Qwen3-VL-8B-NSFW-Caption-V4.5")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))

# Globale Engine
engine = None
serving_chat = None
serving_models = None


async def init_engine():
    global engine, serving_chat, serving_models

    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=True,
    )

    engine = await build_async_engine_client(engine_args)

    serving_models = OpenAIServingModels(
        engine_client=engine,
        model_config=await engine.get_model_config(),
        base_model_paths=[MODEL_NAME],
    )

    serving_chat = OpenAIServingChat(
        engine_client=engine,
        model_config=await engine.get_model_config(),
        models=serving_models,
        request_id="runpod",
    )


async def handler_async(job):
    global engine, serving_chat

    if engine is None:
        await init_engine()

    job_input = job["input"]

    # OpenAI-kompatiblen Route dispatchen
    openai_route = job_input.get("openai_route", "/v1/chat/completions")
    openai_input = job_input.get("openai_input", job_input)

    if openai_route == "/v1/models" or openai_route == "/models":
        models = await serving_models.show_available_models()
        return models.model_dump()

    if openai_route == "/v1/chat/completions" or openai_route == "/chat/completions":
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest

        request = ChatCompletionRequest(**openai_input)
        response = await serving_chat.create_chat_completion(request, raw_request=None)

        if hasattr(response, "model_dump"):
            return response.model_dump()
        return json.loads(response.body)

    return {"error": f"Unknown route: {openai_route}"}


def handler(job):
    return asyncio.get_event_loop().run_until_complete(handler_async(job))


# Pre-load model at startup
print(f"[RunPod Handler] Loading model: {MODEL_NAME}")
asyncio.get_event_loop().run_until_complete(init_engine())
print(f"[RunPod Handler] Model loaded, starting worker")

runpod.serverless.start({"handler": handler})
