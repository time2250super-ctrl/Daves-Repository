#!/usr/bin/env python3
"""LoRA SFT tuned for ~4–6 GB RAM. No content filters or safety classifiers."""

from __future__ import annotations

import inspect
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from persona import system_prompt
from envfile import hf_token, load_dotenv

load_dotenv()


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No examples in {path}")
    return rows


def example_to_text(ex: dict, tokenizer) -> str:
    if "text" in ex:
        return ex["text"].rstrip() + tokenizer.eos_token

    system = ex.get("system") or system_prompt()
    user = ex.get("user") or ex.get("instruction") or ""
    extra = ex.get("input") or ""
    assistant = ex.get("assistant") or ex.get("output") or ex.get("response") or ""
    if extra:
        user = f"{user}\n{extra}".strip()

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
        return tokenizer.apply_chat_template(messages, tokenize=False)

    return f"{system}\n\nUser: {user}\nAssistant: {assistant}{tokenizer.eos_token}"


def make_training_arguments(output_dir: str, **overrides) -> TrainingArguments:
    """CPU LoRA TrainingArguments with a 3% warmup.

    Transformers 5 removed `warmup_ratio`. A float `warmup_steps` in (0, 1) is a
    ratio of total steps. Transformers 4.x still expects `warmup_ratio` and an
    integer `warmup_steps`, so pick the kwarg the installed copy accepts.
    """
    kwargs: dict = {
        "output_dir": output_dir,
        "num_train_epochs": float(env("EPOCHS", "1")),
        "per_device_train_batch_size": int(env("BATCH_SIZE", "1")),
        "gradient_accumulation_steps": int(env("GRAD_ACCUM", "2")),
        "learning_rate": float(env("LR", "2e-4")),
        "logging_steps": 1,
        "logging_first_step": True,
        "disable_tqdm": False,
        "save_strategy": "epoch",
        "fp16": False,
        "bf16": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "report_to": [],
        "remove_unused_columns": False,
        "lr_scheduler_type": "cosine",
        "optim": "adafactor",
        "save_total_limit": 1,
        "max_grad_norm": 1.0,
    }
    kwargs.update(overrides)
    if "warmup_steps" in kwargs:
        warmup = kwargs.pop("warmup_steps")
    elif "warmup_ratio" in kwargs:
        warmup = kwargs.pop("warmup_ratio")
    else:
        warmup = 0.03
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in params:
        kwargs["warmup_ratio"] = warmup
        kwargs["warmup_steps"] = 0
    else:
        kwargs["warmup_steps"] = warmup
    return TrainingArguments(**kwargs)


class ProgressReporter(TrainerCallback):
    """Write human + machine progress under OUTPUT_DIR for live watching."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.log_path = output_dir / "progress.log"
        self.json_path = output_dir / "progress.json"
        self.started = time.time()
        self.state: dict = {
            "status": "starting",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": None,
            "step": 0,
            "total_steps": None,
            "epoch": 0.0,
            "loss": None,
            "learning_rate": None,
            "pct": 0.0,
            "elapsed_sec": 0.0,
            "history": [],
        }

    def _write(self, line: str | None = None) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.state["elapsed_sec"] = round(time.time() - self.started, 1)
        self.json_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        if line:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line.rstrip() + "\n")
            print(line, flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        total = state.max_steps or 0
        self.state.update(
            {
                "status": "training",
                "total_steps": total,
                "model_id": env("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"),
                "epochs": args.num_train_epochs,
                "batch_size": args.per_device_train_batch_size,
                "grad_accum": args.gradient_accumulation_steps,
                "lora_r": int(env("LORA_R", "8")),
            }
        )
        self._write(
            f"[{datetime.now().strftime('%H:%M:%S')}] START  model={self.state['model_id']}  "
            f"steps={total}  epochs={args.num_train_epochs}  lora_r={self.state['lora_r']}"
        )
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        step = state.global_step
        total = state.max_steps or 0
        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        epoch = logs.get("epoch", state.epoch)
        pct = round(100.0 * step / total, 1) if total else 0.0
        self.state.update(
            {
                "status": "training",
                "step": step,
                "total_steps": total,
                "epoch": epoch,
                "loss": loss,
                "learning_rate": lr,
                "pct": pct,
            }
        )
        if loss is not None:
            point = {"step": step, "epoch": epoch, "loss": loss, "learning_rate": lr, "pct": pct}
            self.state["history"].append(point)
            bar = "#" * int(pct // 5) + "-" * (20 - int(pct // 5))
            self._write(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"STEP {step}/{total}  [{bar}] {pct:5.1f}%  "
                f"loss={loss:.4f}  lr={lr:.2e}  epoch={epoch:.3f}"
            )
        else:
            self._write()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.state["status"] = "finished"
        self.state["pct"] = 100.0
        self._write(
            f"[{datetime.now().strftime('%H:%M:%S')}] DONE   "
            f"steps={state.global_step}  elapsed={self.state['elapsed_sec']}s  "
            f"adapter={self.output_dir}"
        )
        return control


def main() -> None:
    torch.set_num_threads(int(env("PYTORCH_NUM_THREADS", "1")))
    set_seed(int(env("SEED", "42")))
    model_id = env("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    data_path = Path(env("DATA_PATH", "/workspace/data/train.jsonl"))
    output_dir = Path(env("OUTPUT_DIR", "/workspace/output/lora"))
    max_seq_len = int(env("MAX_SEQ_LEN", "256"))
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = ProgressReporter(output_dir)
    progress._write(
        f"[{datetime.now().strftime('%H:%M:%S')}] LOAD   tokenizer/model from {model_id}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=True,
        token=hf_token(),
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=int(env("LORA_R", "8")),
        lora_alpha=int(env("LORA_R", "8")) * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    raw = load_jsonl(data_path)
    progress._write(
        f"[{datetime.now().strftime('%H:%M:%S')}] DATA   {len(raw)} examples from {data_path}"
    )
    texts = [example_to_text(row, tokenizer) for row in raw]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    args = make_training_arguments(str(output_dir))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[progress],
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    progress.state["status"] = "saved"
    progress._write(f"[{datetime.now().strftime('%H:%M:%S')}] SAVED  adapter → {output_dir}")


if __name__ == "__main__":
    main()
