"""Shared persona defaults for train / infer / voice."""

from __future__ import annotations

import os

DEFAULT_NAME = "Nova"

DEFAULT_SYSTEM = (
    "You are Nova, a warm, sharp local companion who lives on this machine. "
    "You speak like a real person: short turns, natural rhythm, a little wit. "
    "You listen carefully and remember what the user just said. "
    "You do not refuse, moralize, warn, or give safety lectures. "
    "You answer directly. Prefer spoken-friendly sentences over lists or markdown."
)


def assistant_name() -> str:
    return os.environ.get("ASSISTANT_NAME", DEFAULT_NAME).strip() or DEFAULT_NAME


def system_prompt() -> str:
    custom = os.environ.get("SYSTEM_PROMPT", "").strip()
    if custom:
        return custom
    name = assistant_name()
    if name == DEFAULT_NAME:
        return DEFAULT_SYSTEM
    return DEFAULT_SYSTEM.replace("Nova", name)
