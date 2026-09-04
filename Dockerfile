# Voice + LoRA image (CPU, ~4–6 GB RAM)
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PYTORCH_NUM_THREADS=1 \
    OUTPUT_DIR=/workspace/output/lora \
    ADAPTER_DIR=/workspace/output/lora \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /workspace/requirements.txt

COPY train.py infer.py serve.py persona.py envfile.py /workspace/
COPY static /workspace/static

EXPOSE 7860
ENTRYPOINT ["python", "train.py"]
