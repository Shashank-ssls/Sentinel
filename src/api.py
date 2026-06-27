"""Phase 7 — local demo backend (FastAPI). Fully offline; no external calls.

Loads the two binary models from disk and runs a message through BOTH, so the
UI can show the research finding live: Model W (Western-only) vs Model A
(India-augmented) on the same Indian message. The 3-class production model is
also loaded for richer context, but the W-vs-A binary comparison is the
headline.

Endpoint:
    POST /analyze {"text": "..."} -> per-model verdict + confidence + the top
    contributing features (reusing src.explain.explain — not rebuilt).

The frontend (``frontend/``) is served as static files from the same origin,
so the whole demo runs from one process:

    uvicorn src.api:app --port 8000      # then open http://127.0.0.1:8000

Honest scope: these binary models do **unwanted vs legitimate**, not
smishing-specific detection — the UI copy says so.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import PROJECT_ROOT, get_config
from src.eval_india import build_eval_frame
from src.explain import explain, get_coefficients

FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"
MODELS_DIR: Path = PROJECT_ROOT / "models"
WESTERN_PATH: Path = MODELS_DIR / "western_binary_model.pkl"
INDIA_PATH: Path = MODELS_DIR / "india_augmented_model.pkl"
FIGURES_DIR: Path = PROJECT_ROOT / "paper" / "figures"

# Display names + honest labels.
DISPLAY = {"legit": "legitimate", "unwanted": "unwanted"}
WHY_K = 40  # pull a deep list; the UI shows ~6 then "show all"

# India-vocabulary cues A is expected to learn and W to lack. Distinctive cues
# match as substrings; short/ambiguous ones must match a whole token (so "rs"
# does not fire inside "fi-rs-t").
INDIA_DISTINCT = (
    "airtel",
    "vodafone",
    "aadhaar",
    "fastag",
    "icici",
    "paytm",
    "phonepe",
    "recharge",
    "pantaloons",
    "vishal",
    "jio",
    "kotak",
    "rupees",
    "cashback",
)
INDIA_EXACT = frozenset(
    {"rs", "vi", "inr", "sbi", "hdfc", "axis", "upi", "kyc", "yono", "lakh", "crore"}
)


def _is_india_term(label: str) -> bool:
    t = label.lower()
    if any(d in t for d in INDIA_DISTINCT):
        return True
    return any(tok in INDIA_EXACT for tok in t.split())


class AnalyzeRequest(BaseModel):
    text: str


@lru_cache(maxsize=1)
def get_models() -> dict:
    """Load both binary models + the 3-class production model once."""
    models = {}
    if not WESTERN_PATH.exists() or not INDIA_PATH.exists():
        raise FileNotFoundError(
            "Demo models missing. Run `python -m src.india_augment` first."
        )
    models["W"] = joblib.load(WESTERN_PATH)
    models["A"] = joblib.load(INDIA_PATH)
    prod = get_config().model_path
    models["prod3"] = joblib.load(prod) if Path(prod).exists() else None
    return models


@lru_cache(maxsize=1)
def _w_unwanted_weights() -> dict[str, float]:
    """Map raw feature name -> Model W coefficient toward 'unwanted'."""
    classes, names, coef = get_coefficients(get_models()["W"])
    u = classes.index("unwanted")
    return {n: float(coef[u, i]) for i, n in enumerate(names)}


def _analyze_with(pipe, text: str, name: str, blurb: str) -> dict:
    """Run one model and shape its verdict + full why-list for the UI."""
    exp = explain(text, pipe=pipe, k=WHY_K)
    verdict = exp["predicted"]
    probs = exp["probabilities"]
    why = [
        {
            "feature": c["feature"],
            "kind": c["kind"],
            "contribution": round(c["contribution"], 4),
        }
        for c in exp["top_contributions"]
        if c["contribution"] > 0
    ]
    return {
        "model": name,
        "blurb": blurb,
        "verdict": verdict,
        "verdict_label": DISPLAY.get(verdict, verdict),
        "confidence": round(float(probs.get(verdict, 0.0)), 4),
        "probabilities": {DISPLAY.get(k, k): round(v, 4) for k, v in probs.items()},
        "why": why,
    }


def _india_terms_learned(a_result: dict) -> list[dict]:
    """India tokens A weighted toward 'unwanted' that W scores ~0 or never saw."""
    wmap = _w_unwanted_weights()
    out = []
    for c in a_result["why"]:
        if c["contribution"] <= 0 or not _is_india_term(c["feature"]):
            continue
        prefix = "tfidf__" if c["kind"] == "term" else f"{c['kind']}__"
        w_weight = wmap.get(prefix + c["feature"])
        out.append(
            {
                "feature": c["feature"],
                "kind": c["kind"],
                "a_contribution": c["contribution"],
                "w_weight": None if w_weight is None else round(w_weight, 3),
                "w_status": (
                    "never seen"
                    if w_weight is None
                    else ("~0" if abs(w_weight) < 0.05 else f"{w_weight:+.2f}")
                ),
            }
        )
    return out[:8]


def analyze_message(text: str) -> dict:
    """Core analysis used by the route (and directly by tests)."""
    models = get_models()
    w = _analyze_with(models["W"], text, "Model W", "Western-only (Mendeley)")
    a = _analyze_with(
        models["A"], text, "Model A", "India-augmented (Mendeley + India)"
    )
    differ = w["verdict"] != a["verdict"]
    headline = None
    if differ:
        if w["verdict"] == "legit" and a["verdict"] == "unwanted":
            headline = "Model W missed it. Model A caught it."
        elif w["verdict"] == "unwanted" and a["verdict"] == "legit":
            headline = "Models disagree: W flags, A clears."
        else:
            headline = "Models disagree."
    else:
        headline = (
            "Both models agree: unwanted."
            if w["verdict"] == "unwanted"
            else "Both models agree: legitimate."
        )

    result = {
        "message": text,
        "models": {"W": w, "A": a},
        "verdicts_differ": differ,
        "headline": headline,
    }
    # On the signature case (W clears, A flags), surface the India vocabulary
    # A learned that W lacks — the explanation of why A wins.
    if differ and w["verdict"] == "legit" and a["verdict"] == "unwanted":
        result["india_terms_learned"] = _india_terms_learned(a)
    prod = models.get("prod3")
    if prod is not None:
        p = explain(text, pipe=prod, k=6)
        result["production_3class"] = {
            "verdict": p["predicted"],
            "probabilities": {k: round(v, 4) for k, v in p["probabilities"].items()},
        }
    return result


app = FastAPI(title="Sentinel — local demo", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    models = get_models()
    return {"status": "ok", "models_loaded": [k for k, v in models.items() if v]}


@lru_cache(maxsize=1)
def _load_metrics() -> dict:
    """Aggregate held-out metrics, pulled from paper/figures/ artifacts."""
    aug = json.loads((FIGURES_DIR / "india_augment_results.json").read_text())
    sim = json.loads((FIGURES_DIR / "india_split_similarity_summary.json").read_text())
    res = aug["results"]

    def f1(key: str) -> float:
        return round(res[key]["macro_f1"], 4)

    return {
        "headline": {
            "india_W": f1("W_on_India"),
            "india_A": f1("A_on_India"),
            "india_gain": round(aug["headline"]["india_gain"], 4),
            "mendeley_W": f1("W_on_Mendeley"),
            "mendeley_A": f1("A_on_Mendeley"),
            "western_delta": round(aug["headline"]["western_delta"], 4),
            "novel_bucket_f1": round(sim["model_A_macro_f1"]["novel"]["macro_f1"], 4),
            "novel_bucket_n": sim["model_A_macro_f1"]["novel"]["n"],
            "newly_caught": aug["headline"]["n_newly_caught"],
        },
        "comparison": [
            {
                "model": "W (Western-only)",
                "mendeley": f1("W_on_Mendeley"),
                "india": f1("W_on_India"),
            },
            {
                "model": "A (India-augmented)",
                "mendeley": f1("A_on_Mendeley"),
                "india": f1("A_on_India"),
            },
        ],
        "cv": {
            "W": {k: round(v, 4) for k, v in aug["cv"]["W"].items()},
            "A": {k: round(v, 4) for k, v in aug["cv"]["A"].items()},
        },
        "sizes": aug["sizes"],
        "diversity_pct_unique": sim["diversity"]["pct_unique"],
    }


@app.get("/metrics")
def metrics() -> dict:
    try:
        return _load_metrics()
    except FileNotFoundError:
        return {
            "error": "metrics artifacts not found; run src.india_augment + src.diag_india_split"
        }


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    return analyze_message(req.text)


# Serve the single-page frontend from the same origin (mounted last so the
# API routes above take precedence).
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
