# Local uncensored training + voice companion (Docker, low RAM)

LoRA trainer with **no** Llama-Guard, dataset filter, or refusal post-processor. The default model (`Qwen/Qwen2.5-1.5B-Instruct`) holds a real conversation and needs about **8 GB RAM** on CPU. For the **4–6 GB** profile, set `MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct` in `.env`.

Default persona: **Nova** — warm, direct, spoken-friendly. Rename with `ASSISTANT_NAME` in `.env`.

## Voice (mic → her ears, speakers → her voice)

Mic and speakers stay in **Chrome/Edge**. The model runs in Docker.

```powershell
copy .env.example .env
```

Set `HF_TOKEN` in `.env` (Hugging Face read token) so downloads are authenticated. Then:

```powershell
docker compose build voice
docker compose up voice
```

Open **http://127.0.0.1:7860/**  
Allow the microphone. Hold **Hold to talk**, release to send. She answers with browser speech synthesis (picks a female system voice when available).

**Teach this turn** appends that exchange into `data/train.jsonl`. Then re-run LoRA so she learns from your voice chats:

```powershell
docker compose run --rm lora
```

All voice/text turns are also logged to `data/voice_memory.jsonl`.

## LoRA image

`Dockerfile.lora` → `local-ai-lora:latest`. Progress:

- `output/lora/progress.log`
- `output/lora/progress.json`

```powershell
docker compose build lora
docker compose run --rm lora
.\watch-progress.ps1
```

## Text chat

```powershell
.\chat.ps1
```

## Data

`data/train.jsonl`, one line per example:

```json
{"user": "...", "assistant": "..."}
```

## What will not fit in 4–6 GB

Dolphin / Hermes 8B, QLoRA 7B, and `nvidia/cuda` devel images.
