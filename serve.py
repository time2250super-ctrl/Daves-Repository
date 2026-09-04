#!/usr/bin/env python3
"""Voice + text chat server. Mic/speakers stay in the browser; model runs here."""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_from_directory
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona import assistant_name, system_prompt
from envfile import hf_token, load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
MEMORY = Path(os.environ.get("VOICE_MEMORY", str(ROOT / "data" / "voice_memory.jsonl")))
TEACH = Path(os.environ.get("TEACH_PATH", str(ROOT / "data" / "train.jsonl")))

app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
_lock = threading.Lock()
_tokenizer = None
_model = None

# Per-client conversation history keyed by a browser-supplied session id, so
# multiple tabs/clients do not share or clobber each other's context.
_MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "64"))
_histories: "OrderedDict[str, list[dict]]" = OrderedDict()


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def load_model() -> None:
    global _tokenizer, _model
    torch.set_num_threads(int(env("PYTORCH_NUM_THREADS", "1")))
    model_id = env("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_dir = env("ADAPTER_DIR", str(ROOT / "output" / "lora"))

    print(f"Loading {model_id} …", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token()
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=True,
        token=hf_token(),
    )
    if Path(adapter_dir, "adapter_config.json").is_file():
        _model = PeftModel.from_pretrained(_model, adapter_dir)
        print(f"Loaded adapter {adapter_dir}", flush=True)
    else:
        print("No adapter; base model only.", flush=True)
    _model.eval()
    print(f"Ready as {assistant_name()}", flush=True)


def _get_history(session_id: str) -> list[dict]:
    """Return (creating if needed) the history for a session, LRU-capped."""
    history = _histories.get(session_id)
    if history is None:
        history = [{"role": "system", "content": system_prompt()}]
        _histories[session_id] = history
        while len(_histories) > _MAX_SESSIONS:
            _histories.popitem(last=False)
    else:
        _histories.move_to_end(session_id)
    return history


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate(user_text: str, session_id: str) -> str:
    assert _tokenizer is not None and _model is not None
    max_new = int(env("MAX_NEW_TOKENS", "96"))

    with _lock:
        history = _get_history(session_id)
        history.append({"role": "user", "content": user_text})
        if hasattr(_tokenizer, "apply_chat_template") and _tokenizer.chat_template:
            prompt = _tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = (
                system_prompt()
                + "\n\n"
                + "\n".join(
                    f"{m['role']}: {m['content']}" for m in history if m["role"] != "system"
                )
                + "\nassistant:"
            )
        inputs = _tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = _model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.85,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        text = _tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        # Keep spoken replies short and clean
        text = text.split("\n")[0].strip()
        for stop in ("User:", "user:", "You:", "System:"):
            if stop in text:
                text = text.split(stop)[0].strip()
        history.append({"role": "assistant", "content": text})
        if len(history) > 9:
            del history[1:-8]
        return text


@app.get("/")
def index():
    return send_from_directory(STATIC, "voice.html")


@app.get("/api/persona")
def persona_info():
    return jsonify(
        {
            "name": assistant_name(),
            "system_prompt": system_prompt(),
            "voice_hint": env("BROWSER_VOICE_HINT", "female"),
        }
    )


def _session_id(body: dict) -> str:
    return (body.get("session") or "").strip() or "default"


@app.post("/api/chat")
def chat():
    body = request.get_json(force=True, silent=True) or {}
    user_text = (body.get("text") or "").strip()
    source = (body.get("source") or "text").strip()
    if not user_text:
        return jsonify({"error": "empty"}), 400
    reply = generate(user_text, _session_id(body))
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "user": user_text,
        "assistant": reply,
        "system": system_prompt(),
    }
    append_jsonl(MEMORY, row)
    return jsonify({"name": assistant_name(), "reply": reply})


@app.post("/api/teach")
def teach():
    """Save the last voice/text turn into train.jsonl for the next LoRA run."""
    body = request.get_json(force=True, silent=True) or {}
    user_text = (body.get("user") or "").strip()
    assistant_text = (body.get("assistant") or "").strip()
    if not user_text or not assistant_text:
        return jsonify({"error": "need user and assistant"}), 400
    row = {
        "system": system_prompt(),
        "user": user_text,
        "assistant": assistant_text,
    }
    append_jsonl(TEACH, row)
    return jsonify({"ok": True, "path": str(TEACH)})


@app.post("/api/reset")
def reset():
    body = request.get_json(force=True, silent=True) or {}
    with _lock:
        _histories.pop(_session_id(body), None)
    return jsonify({"ok": True})


def main() -> None:
    load_model()
    host = env("HOST", "0.0.0.0")
    port = int(env("PORT", "7860"))
    print(f"Open http://127.0.0.1:{port}/  (mic + speakers in the browser)", flush=True)
    # Prefer the production-grade waitress server; fall back to Flask's
    # development server only if waitress is unavailable.
    try:
        from waitress import serve as _serve
    except ImportError:
        app.run(host=host, port=port, threaded=True)
    else:
        _serve(app, host=host, port=port, threads=int(env("SERVER_THREADS", "4")))


if __name__ == "__main__":
    main()
