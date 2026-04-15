"""RunPod Serverless Handler - startet vLLM Server und proxied Requests."""
import os
import subprocess
import time
import requests
import runpod

MODEL_NAME = os.getenv("MODEL_NAME", "monkeyslikebananas/Qwen3-VL-8B-NSFW-Caption-V4.5")
MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "4096")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.90")
VLLM_PORT = 8000
VLLM_URL = f"http://localhost:{VLLM_PORT}"

# vLLM Server als subprocess starten
print(f"[Handler] Starting vLLM server with model: {MODEL_NAME}")
vllm_proc = subprocess.Popen([
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_NAME,
    "--max-model-len", MAX_MODEL_LEN,
    "--gpu-memory-utilization", GPU_MEMORY_UTILIZATION,
    "--trust-remote-code",
    "--host", "0.0.0.0",
    "--port", str(VLLM_PORT),
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# Warten bis Server ready ist
print("[Handler] Waiting for vLLM server to start...")
for i in range(300):
    try:
        r = requests.get(f"{VLLM_URL}/health", timeout=2)
        if r.status_code == 200:
            print(f"[Handler] vLLM server ready after {i}s")
            break
    except Exception:
        pass
    time.sleep(1)
else:
    print("[Handler] ERROR: vLLM server failed to start")


def handler(job):
    job_input = job["input"]
    openai_route = job_input.get("openai_route", "/v1/chat/completions")
    openai_input = job_input.get("openai_input", job_input)

    try:
        if "models" in openai_route:
            r = requests.get(f"{VLLM_URL}{openai_route}", timeout=30)
        else:
            r = requests.post(
                f"{VLLM_URL}{openai_route}",
                json=openai_input,
                timeout=120,
            )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


print("[Handler] Starting RunPod worker")
runpod.serverless.start({"handler": handler})
