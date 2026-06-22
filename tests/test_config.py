"""Tests for src.config — the environment-backed configuration loader."""

from pathlib import Path

from src.config import Config, PROJECT_ROOT, get_config


def test_defaults_apply_when_env_absent(monkeypatch):
    """With no env vars set, documented defaults are used."""
    for var in ("MODEL_PATH", "CACHE_DB", "RANDOM_SEED"):
        monkeypatch.delenv(var, raising=False)

    cfg = Config.from_env()

    assert cfg.model_path == PROJECT_ROOT / "models" / "best_model.pkl"
    assert cfg.cache_db == PROJECT_ROOT / "data" / "cache.db"
    assert cfg.random_seed == 42


def test_env_overrides_are_honored(monkeypatch):
    """Environment variables override the defaults."""
    monkeypatch.setenv("RANDOM_SEED", "1234")
    monkeypatch.setenv("MODEL_PATH", "models/custom.pkl")

    cfg = Config.from_env()

    assert cfg.random_seed == 1234
    assert cfg.model_path == PROJECT_ROOT / "models" / "custom.pkl"


def test_absolute_paths_are_preserved(monkeypatch, tmp_path):
    """An absolute path in the env is used as-is, not re-rooted."""
    abs_model = tmp_path / "m.pkl"
    monkeypatch.setenv("MODEL_PATH", str(abs_model))

    cfg = Config.from_env()

    assert cfg.model_path == abs_model


def test_get_config_returns_config_instance():
    assert isinstance(get_config(), Config)
