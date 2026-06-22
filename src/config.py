"""Central configuration, read from environment / ``.env``.

All tunable paths and the random seed live here so that training and live
inference read the *same* values (no drift) and runs stay reproducible.
Secrets and machine-specific paths belong in ``.env`` (never hard-coded);
see ``.env.example`` for the contract.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of this file's directory (src/ -> repo root).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Load .env once at import time. Values already in the environment win, so
# CI / shell overrides are respected. Missing .env is fine (defaults apply).
load_dotenv(PROJECT_ROOT / ".env")


def _resolve(path_str: str) -> Path:
    """Resolve a possibly-relative config path against the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration for Sentinel."""

    model_path: Path
    cache_db: Path
    random_seed: int

    @classmethod
    def from_env(cls) -> "Config":
        """Build a :class:`Config` from environment variables with defaults."""
        return cls(
            model_path=_resolve(os.getenv("MODEL_PATH", "models/best_model.pkl")),
            cache_db=_resolve(os.getenv("CACHE_DB", "data/cache.db")),
            random_seed=int(os.getenv("RANDOM_SEED", "42")),
        )


def get_config() -> Config:
    """Return the active configuration loaded from the environment."""
    return Config.from_env()


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    cfg = get_config()
    print("Sentinel config:")
    print(f"  PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"  model_path   = {cfg.model_path}")
    print(f"  cache_db     = {cfg.cache_db}")
    print(f"  random_seed  = {cfg.random_seed}")
