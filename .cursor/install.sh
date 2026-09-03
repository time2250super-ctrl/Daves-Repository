#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the Nova voice companion + LoRA trainer.
#
# The personal environment install command is `bash .cursor/install.sh`.
# Cloud Agent VMs have no NVIDIA GPU, so this provisions a CPU Python venv.
set -euo pipefail

cd "$(dirname "$0")/.."

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

echo "Install complete. Activate with: source ${VENV_DIR}/bin/activate"
