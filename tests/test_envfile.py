"""Tests for .env loading and HF token mirroring."""

from __future__ import annotations

import os

import envfile


def test_load_dotenv_sets_missing_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("ASSISTANT_NAME", raising=False)
    path = tmp_path / ".env"
    path.write_text("HF_TOKEN=hf_test_token\nASSISTANT_NAME=Ada\n", encoding="utf-8")
    envfile.load_dotenv(path)
    assert os.environ["HF_TOKEN"] == "hf_test_token"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_test_token"
    assert os.environ["ASSISTANT_NAME"] == "Ada"
    assert envfile.hf_token() == "hf_test_token"


def test_load_dotenv_does_not_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "already-set")
    path = tmp_path / ".env"
    path.write_text("HF_TOKEN=from-file\n", encoding="utf-8")
    envfile.load_dotenv(path)
    assert os.environ["HF_TOKEN"] == "already-set"


def test_hf_token_none_when_empty(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert envfile.hf_token() is None
