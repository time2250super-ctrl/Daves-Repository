#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the Nova voice companion + LoRA trainer.
#
# The environment install command is `bash .cursor/install.sh`.
# Cloud Agent VMs have no NVIDIA GPU, so this provisions a CPU Python venv.
# (The Docker images in this repo are also CPU-only; a GPU host needs a custom
# image with a CUDA build of PyTorch.)
set -euo pipefail

cd "$(dirname "$0")/.."

# Local .env (never printed). Existing exported vars win.
if [ -f .env ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            ''|\#*) continue ;;
        esac
        key="${line%%=*}"
        val="${line#*=}"
        key="${key%"${key##*[![:space:]]}"}"
        key="${key#"${key%%[![:space:]]*}"}"
        [ -z "$key" ] && continue
        if [ -z "${!key+x}" ]; then
            export "$key=$val"
        fi
    done < .env
fi
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

# `python -m venv --help` succeeds even when the python3-venv package
# (ensurepip) is missing, so probe ensurepip directly.
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

# CPU build of PyTorch (no CUDA on Cloud Agent VMs).
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

python -c "import torch, transformers, peft, datasets, accelerate, flask; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| flask', flask.__version__)"

# Pre-cache the default chat model so the voice server starts quickly on boot.
# Non-fatal: if the download is unavailable, the server fetches it on first run.
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}" python - <<'PY' || echo "Model prefetch skipped; it will download on first server start."
import os
from huggingface_hub import snapshot_download
model_id = os.environ["MODEL_ID"]
token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None
if token:
    print(f"Prefetching {model_id} (authenticated) …", flush=True)
else:
    print(f"Prefetching {model_id} anonymously. Set HF_TOKEN in .env for higher rate limits.", flush=True)
snapshot_download(model_id, token=token)
print("Prefetch complete.", flush=True)
PY

echo "Install complete. Activate with: source ${VENV_DIR}/bin/activate"
