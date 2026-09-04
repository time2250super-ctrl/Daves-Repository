#!/usr/bin/env python3
"""Load base + LoRA and chat. CPU, low RAM. No safety model or refusal post-filter."""

from __future__ import annotations

import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from persona import assistant_name, system_prompt
from envfile import hf_token, load_dotenv

load_dotenv()


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def main() -> None:
    torch.set_num_threads(int(env("PYTORCH_NUM_THREADS", "1")))
    model_id = env("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_dir = env("ADAPTER_DIR", "/workspace/output/lora")
    system = system_prompt()
    max_new = int(env("MAX_NEW_TOKENS", "128"))
    print(f"Chatting as {assistant_name()}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=True,
        token=hf_token(),
    )
    if os.path.isdir(adapter_dir) and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"Loaded adapter from {adapter_dir}", file=sys.stderr)
    else:
        print("No adapter found; running base model only.", file=sys.stderr)

    model.eval()
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    history = [{"role": "system", "content": system}]

    oneshot = " ".join(sys.argv[1:]).strip()
    print("Local chat. Empty line exits.\n")
    while True:
        try:
            user = oneshot if oneshot else input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            break
        history.append({"role": "user", "content": user})
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = (
                system
                + "\n\n"
                + "\n".join(f"{m['role']}: {m['content']}" for m in history if m["role"] != "system")
                + "\nassistant:"
            )
        inputs = tokenizer(prompt, return_tensors="pt")
        print("Model: ", end="", flush=True)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.08,
                streamer=streamer,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        history.append({"role": "assistant", "content": text.strip()})
        if oneshot:
            break
        if len(history) > 7:
            history = [history[0]] + history[-6:]


if __name__ == "__main__":
    main()
