"""Tests for src.api — the local demo backend (both-model verdicts)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import analyze_message, app
from src.config import PROJECT_ROOT

MODELS_READY = (PROJECT_ROOT / "models" / "western_binary_model.pkl").exists() and (
    PROJECT_ROOT / "models" / "india_augmented_model.pkl"
).exists()
needs_models = pytest.mark.skipif(
    not MODELS_READY, reason="demo models not built (run src.india_augment)"
)

SAMPLE = (
    "ICICI Alert: A UPI collect request of Rs 18,450 is pending. "
    "Verify immediately at icici-secure-review.in"
)


@needs_models
def test_analyze_returns_both_models():
    out = analyze_message(SAMPLE)
    assert set(out["models"]) == {"W", "A"}
    for m in out["models"].values():
        assert m["verdict"] in {"legit", "unwanted"}
        assert 0.0 <= m["confidence"] <= 1.0
        assert isinstance(m["why"], list)
    assert isinstance(out["verdicts_differ"], bool)
    assert out["headline"]


@needs_models
def test_endpoint_analyze():
    client = TestClient(app)
    resp = client.post("/analyze", json={"text": SAMPLE})
    assert resp.status_code == 200
    data = resp.json()
    assert "W" in data["models"] and "A" in data["models"]


@needs_models
def test_health_ok():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@needs_models
def test_app_serves_frontend_index():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "SENTINEL" in resp.text


def test_preloaded_examples_are_valid():
    """The frontend's preloaded examples must be well-formed and non-empty."""
    app_js = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "EXAMPLES" in app_js
    # Each example carries a label and a text field.
    assert app_js.count("label:") == app_js.count("text:")
    assert app_js.count("label:") >= 5  # scams + promo disagreements + legit
    # Curated set: a clear scam, the real W-miss/A-catch promos, and legit.
    for kw in ("UPI", "KYC", "Airtel", "Vi"):
        assert kw in app_js
