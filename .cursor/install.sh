#!/usr/bin/env bash
# Idempotent dependency setup for the Nova voice companion + LoRA trainer.
#
# Cloud Agent VMs have no NVIDIA GPU, so this sets up a CPU-capable Python
# environment. The app (train.py / infer.py / serve.py) is CPU-friendly and
# runs on the small default instruct model out of the box.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

# Ensure venv can bootstrap pip. `python -m venv --help` succeeds even when the
# python3-venv package (which provides ensurepip) is missing, so probe ensurepip
# directly and install the matching venv package when it is absent.
if ! "${PYTHON_BIN}" -c 'import ensurepip' >/dev/null 2>&1; then
    PY_MM="$("${PYTHON_BIN}" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update -qq
        sudo apt-get install -y "python${PY_MM}-venv" python3-venv
    else
        apt-get update -qq
        apt-get install -y "python${PY_MM}-venv" python3-venv
    fi
fi

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    rm -rf "${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

# CPU build of PyTorch (no CUDA on Cloud Agent VMs). On a GPU host the
# Dockerfile installs the CUDA build instead.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

python -c "import torch, transformers, peft, datasets, accelerate, flask; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| flask', flask.__version__)"

# Pre-cache the default chat model so the voice server starts quickly on boot.
# Non-fatal: if the download is unavailable, the server fetches it on first run.
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}" python - <<'PY' || echo "Model prefetch skipped; it will download on first server start."
import os
from huggingface_hub import snapshot_download
model_id = os.environ["MODEL_ID"]
print(f"Prefetching {model_id} …", flush=True)
snapshot_download(model_id)
print("Prefetch complete.", flush=True)
PY

echo "Install complete. Activate with: source ${VENV_DIR}/bin/activate"
