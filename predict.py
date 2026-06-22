"""CLI: classify a single SMS message with the trained Stage-1 model.

Usage::

    python predict.py "Your KYC is blocked, verify now at http://bit.ly/x"

Loads the fitted pipeline from ``config.model_path`` (built by ``src.train``)
and prints the predicted class plus per-class probabilities.
"""

from __future__ import annotations

import sys

import joblib

from src.config import get_config
from src.features import message_to_frame


def predict(text: str) -> tuple[str, dict[str, float]]:
    """Return (predicted_class, {class: probability}) for one message."""
    cfg = get_config()
    if not cfg.model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {cfg.model_path}. Run `python -m src.train` first."
        )
    pipe = joblib.load(cfg.model_path)
    X = message_to_frame(text)
    pred = str(pipe.predict(X)[0])
    proba = pipe.predict_proba(X)[0]
    probs = {str(c): float(p) for c, p in zip(pipe.classes_, proba)}
    return pred, probs


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print('Usage: python predict.py "<message text>"')
        raise SystemExit(2)
    text = sys.argv[1]
    pred, probs = predict(text)
    print(f"message:   {text}")
    print(f"predicted: {pred}")
    print("probabilities:")
    for cls, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {cls:<10} {p:6.3f}")


if __name__ == "__main__":
    main()
