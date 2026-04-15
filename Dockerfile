FROM vllm/vllm-openai:v0.19.0

RUN pip install --no-cache-dir runpod

COPY handler.py /handler.py

ENV MODEL_NAME="monkeyslikebananas/Qwen3-VL-8B-NSFW-Caption-V4.5"
ENV MAX_MODEL_LEN="4096"
ENV GPU_MEMORY_UTILIZATION="0.90"

CMD ["python", "/handler.py"]
