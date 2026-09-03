"""Fast, model-free tests for Nova's persona, session memory, and data formatting."""
from __future__ import annotations

import importlib

import pytest


# --- persona -----------------------------------------------------------------

def test_persona_defaults(monkeypatch):
    monkeypatch.delenv("ASSISTANT_NAME", raising=False)
    monkeypatch.delenv("SYSTEM_PROMPT", raising=False)
    import persona
    assert persona.assistant_name() == "Nova"
    assert persona.system_prompt().startswith("You are Nova")


def test_persona_rename_rewrites_prompt(monkeypatch):
    monkeypatch.setenv("ASSISTANT_NAME", "Ada")
    monkeypatch.delenv("SYSTEM_PROMPT", raising=False)
    import persona
    assert persona.assistant_name() == "Ada"
    prompt = persona.system_prompt()
    assert "Ada" in prompt and "Nova" not in prompt


def test_persona_custom_prompt_wins(monkeypatch):
    monkeypatch.setenv("SYSTEM_PROMPT", "Be terse.")
    import persona
    assert persona.system_prompt() == "Be terse."


# --- serve: per-session history ---------------------------------------------

@pytest.fixture()
def serve_mod(monkeypatch):
    monkeypatch.delenv("ASSISTANT_NAME", raising=False)
    monkeypatch.delenv("SYSTEM_PROMPT", raising=False)
    serve = importlib.import_module("serve")
    serve._histories.clear()
    return serve


def test_new_session_starts_with_system_only(serve_mod):
    h = serve_mod._get_history("A")
    assert len(h) == 1 and h[0]["role"] == "system"


def test_same_session_returns_same_list(serve_mod):
    a1 = serve_mod._get_history("A")
    a1.append({"role": "user", "content": "hi"})
    a2 = serve_mod._get_history("A")
    assert a2 is a1 and len(a2) == 2


def test_sessions_are_isolated(serve_mod):
    a = serve_mod._get_history("A")
    a.append({"role": "user", "content": "my name is Dave"})
    b = serve_mod._get_history("B")
    assert all(m["content"] != "my name is Dave" for m in b)


def test_history_lru_cap(monkeypatch, serve_mod):
    monkeypatch.setattr(serve_mod, "_MAX_SESSIONS", 3)
    for sid in ("s1", "s2", "s3", "s4"):
        serve_mod._get_history(sid)
    assert "s1" not in serve_mod._histories
    assert set(serve_mod._histories) == {"s2", "s3", "s4"}


# --- serve: HTTP endpoints that do not need the model ------------------------

@pytest.fixture()
def client(serve_mod):
    serve_mod.app.config.update(TESTING=True)
    return serve_mod.app.test_client()


def test_persona_endpoint(client):
    resp = client.get("/api/persona")
    assert resp.status_code == 200
    assert resp.get_json()["name"] == "Nova"


def test_chat_rejects_empty_text(client):
    resp = client.post("/api/chat", json={"text": "  ", "session": "A"})
    assert resp.status_code == 400


def test_reset_clears_only_its_session(serve_mod, client):
    serve_mod._get_history("A").append({"role": "user", "content": "remember me"})
    serve_mod._get_history("B").append({"role": "user", "content": "and me"})
    resp = client.post("/api/reset", json={"session": "A"})
    assert resp.status_code == 200 and resp.get_json()["ok"] is True
    assert "A" not in serve_mod._histories
    assert "B" in serve_mod._histories


# --- train: example formatting ----------------------------------------------

class _FakeTokenizer:
    def __init__(self, with_template):
        self.eos_token = "</s>"
        self.chat_template = "TEMPLATE" if with_template else None

    def apply_chat_template(self, messages, tokenize=False):
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


def test_example_to_text_plain_fallback():
    import train
    tok = _FakeTokenizer(with_template=False)
    out = train.example_to_text({"user": "hi", "assistant": "hello"}, tok)
    assert "User: hi" in out and "Assistant: hello" in out
    assert out.endswith("</s>")


def test_example_to_text_uses_chat_template():
    import train
    tok = _FakeTokenizer(with_template=True)
    out = train.example_to_text(
        {"system": "sys", "user": "hi", "assistant": "hello"}, tok
    )
    assert out == "system:sys|user:hi|assistant:hello"


def test_example_to_text_raw_text_passthrough():
    import train
    tok = _FakeTokenizer(with_template=True)
    out = train.example_to_text({"text": "already formatted"}, tok)
    assert out == "already formatted</s>"
