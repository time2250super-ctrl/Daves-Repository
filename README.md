# Local uncensored training (Docker)

This is a **local** LoRA/QLoRA trainer. There is no Llama-Guard, no dataset filter, and no refusal post-processor. It will train on whatever JSONL you put in `data/`. That also means you are responsible for the data and for staying inside the law.

## What you need

- Docker Desktop on Windows with WSL2
- NVIDIA GPU + current Game Ready/Studio driver
- NVIDIA Container Toolkit (Docker Desktop: enable GPU in Settings → Resources → WSL Integration)

Check the GPU is visible from Docker:

```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Data format

`data/train.jsonl`, one example per line:

```json
{"user": "...", "assistant": "..."}
```

Optional keys: `system`, `input`, or a raw `text` field if you want to skip chat templating.

Replace the sample file with your own corpus. Do not mix in safety / refusal pairs if you want an uncensored adapter.

## Train

From `local-ai/`:

```powershell
copy .env.example .env
docker compose build
docker compose run --rm train
```

Adapter lands in `output/lora`.

Swap the base checkpoint in `.env`. `Qwen/Qwen2.5-0.5B` is a VRAM smoke test. For an already-uncensored instruct start, use a Dolphin or Hermes 8B if you have ~10–16 GB VRAM with QLoRA.

Gated Hugging Face models need `HF_TOKEN` in `.env`.

## Chat

```powershell
docker compose --profile infer run --rm infer
```

## What this does not do

- It does not strip safety from a closed API (OpenAI, Claude, etc.).
- It does not magically “remove all alignment” from a heavily RLHF’d checkpoint; for that, start from a **base** model or an already-uncensored instruct (Dolphin / Hermes) and train on unrestricted data.
- Full-parameter training of 7B+ is a different (much larger) GPU/setup problem; this stack is LoRA/QLoRA.
