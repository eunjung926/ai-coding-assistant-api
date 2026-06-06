# syntax=docker/dockerfile:1.4
# GPU inference server. Base model is baked into the image at build time.
# Adapter weights must exist in adapters/vul/ and adapters/ske/.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    BASE_MODEL_PATH=/app/baked/base-model \
    VUL_ADAPTER_PATH=/app/adapters/vul \
    SKE_ADAPTER_PATH=/app/adapters/ske

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

ARG HF_TOKEN
RUN --mount=type=cache,target=/root/.cache/huggingface \
    HF_TOKEN="${HF_TOKEN}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

out = "/app/baked/base-model"
os.makedirs(out, exist_ok=True)
token = os.environ.get("HF_TOKEN") or None
snapshot_download(
    repo_id="Qwen/Qwen2-7B",
    local_dir=out,
    token=token,
)
print("snapshot_download ok ->", out)
PY

COPY . .

EXPOSE 5059

CMD ["python", "run.py"]
