# Setup on a new machine

## 1. Clone

```powershell
git clone https://github.com/time2250super-ctrl/Daves-Repository.git local-ai
cd local-ai
```

## 2. Configure

```powershell
copy .env.example .env
```

Edit `.env` if you want a different name than Nova or a custom `SYSTEM_PROMPT`.
Paste a Hugging Face read token into `HF_TOKEN` so model downloads are authenticated
(https://huggingface.co/settings/tokens). Leave it blank to download anonymously.

## 3. Docker

Requires Docker Desktop (WSL2 on Windows) with enough RAM (~8 GB for the default 1.5B model; ~4–6 GB if you set `MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct`).

```powershell
docker compose build lora voice
```

## 4. Train (optional)

```powershell
docker compose run --rm lora
.\watch-progress.ps1
```

Progress: `output/lora/progress.log`

## 5. Voice chat

```powershell
.\voice.ps1
```

Or: `docker compose up -d voice` then open http://127.0.0.1:7860/

## 6. Text chat

```powershell
.\chat.ps1
```

## What is not in git

- `.env` (secrets / local config)
- `output/` (adapters, progress logs)
- `data/voice_memory.jsonl` (runtime chat log)

After clone, run at least one LoRA train or copy your `output/lora` folder from the old machine if you want the same adapter without retraining.
