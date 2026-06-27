"""Read-only diagnostic — false positives (legit predicted 'unwanted').

Quantifies, for BOTH Model W and Model A, on BOTH held-out sets (India + the
Mendeley test split), how often a truly-legit message is flagged 'unwanted',
the confidence of those errors, the feature profile that drives them, and a
message-type bucketing. Trains/changes NOTHING.

Run ``python -m src.diag_false_positives``. Output: console report +
``paper/figures/false_positive_analysis.csv``.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.data import load_processed
from src.eval_india import _ascii, build_eval_frame, to_unwanted_legit
from src.explain import explain
from src.features import STRUCT_FEATURE_NAMES, compute_structural_features
from src.india_augment import INDIA_AUG_MODEL_PATH, INDIA_TEST_PATH
from src.train import MODELS_DIR

WESTERN_PATH: Path = MODELS_DIR / "western_binary_model.pkl"
FIGURES_DIR: Path = PROJECT_ROOT / "paper" / "figures"
PROFILE_FEATURES = [
    "digit_count",
    "count_currency",
    "message_length",
    "urgency_count",
    "count_links",
    "has_url",
    "uppercase_ratio",
]

# Heuristic message-type buckets for legit messages.
TYPE_RULES = {
    "otp": ("otp", "one time password", "verification code", "is your code"),
    "bank/transactional": (
        "debited",
        "credited",
        "a/c",
        "account",
        "balance",
        "txn",
        "transaction",
        "upi",
        "imps",
        "neft",
        "bill",
        "due",
        "payment",
        "statement",
        "emi",
    ),
    "delivery/logistics": (
        "delivered",
        "out for delivery",
        "shipped",
        "order",
        "courier",
        "parcel",
        "tracking",
        "arriving",
    ),
    "recharge/plan": ("recharge", "plan", "data", "validity", "pack", "gb"),
    "appointment/reminder": (
        "appointment",
        "reminder",
        "scheduled",
        "meeting",
        "booking",
        "slot",
    ),
    "otp_misc": (),
}


def _bucket(text: str) -> str:
    t = text.lower()
    for name, kws in TYPE_RULES.items():
        if kws and any(k in t for k in kws):
            return name
    return "other"


def _eval_model(
    name: str, pipe, texts, true_bin, dataset: str
) -> tuple[dict, pd.DataFrame]:
    """Return summary + per-message frame for legit messages of one model/set."""
    X = build_eval_frame(texts)
    proba = pipe.predict_proba(X)
    classes = list(pipe.classes_)
    u = classes.index("unwanted")
    pred = np.array([classes[i] for i in proba.argmax(1)])
    p_unwanted = proba[:, u]

    struct = compute_structural_features(X)
    sidx = {n: i for i, n in enumerate(STRUCT_FEATURE_NAMES)}

    legit_mask = true_bin == "legit"
    n_legit = int(legit_mask.sum())
    fp_mask = legit_mask & (pred == "unwanted")
    n_fp = int(fp_mask.sum())

    rows = []
    for i in np.where(legit_mask)[0]:
        is_fp = pred[i] == "unwanted"
        rows.append(
            {
                "dataset": dataset,
                "model": name,
                "text": _ascii(texts[i], 220),
                "is_false_positive": bool(is_fp),
                "p_unwanted": round(float(p_unwanted[i]), 4),
                "msg_type": _bucket(texts[i]) if is_fp else _bucket(texts[i]),
                **{f: round(float(struct[i, sidx[f]]), 3) for f in PROFILE_FEATURES},
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "dataset": dataset,
        "model": name,
        "n_legit": n_legit,
        "n_false_positive": n_fp,
        "fp_rate": round(n_fp / n_legit, 4) if n_legit else 0.0,
    }
    return summary, frame


def _conf_bins(p: pd.Series) -> dict:
    if p.empty:
        return {}
    return {
        "mean_p_unwanted": round(p.mean(), 3),
        "median": round(p.median(), 3),
        ">=0.9": int((p >= 0.9).sum()),
        "0.7-0.9": int(((p >= 0.7) & (p < 0.9)).sum()),
        "0.5-0.7": int(((p >= 0.5) & (p < 0.7)).sum()),
    }


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    W = joblib.load(WESTERN_PATH)
    A = joblib.load(INDIA_AUG_MODEL_PATH)

    india = pd.read_csv(INDIA_TEST_PATH)
    india_texts = india["text"].astype(str).tolist()
    india_true = to_unwanted_legit(india["label"])

    _, mend_test = load_processed()
    mend_texts = mend_test["text"].astype(str).tolist()
    mend_true = to_unwanted_legit(mend_test["label"])

    jobs = [
        ("W", W, india_texts, india_true, "india"),
        ("A", A, india_texts, india_true, "india"),
        ("W", W, mend_texts, mend_true, "mendeley"),
        ("A", A, mend_texts, mend_true, "mendeley"),
    ]
    summaries, frames = [], []
    for name, pipe, texts, true_bin, ds in jobs:
        s, f = _eval_model(name, pipe, texts, true_bin, ds)
        summaries.append(s)
        frames.append(f)
    all_legit = pd.concat(frames, ignore_index=True)

    print("=" * 78)
    print("False-positive analysis (legit -> 'unwanted'), held-out sets [read-only]")
    print("=" * 78)
    sm = pd.DataFrame(summaries)
    print(sm.to_string(index=False))

    # Confidence distribution of the false positives.
    print("\nConfidence (P(unwanted)) of false positives:")
    for ds in ("india", "mendeley"):
        for m in ("W", "A"):
            fp = all_legit[
                (all_legit.dataset == ds)
                & (all_legit.model == m)
                & all_legit.is_false_positive
            ]
            print(f"  {ds:9} {m}: n={len(fp):3}  {_conf_bins(fp['p_unwanted'])}")

    # Feature profile: FP legit vs correctly-cleared legit (India, Model A — the
    # production-relevant case; the demo uses A).
    print("\n" + "=" * 78)
    print("Feature profile — India test, Model A: FP legit vs cleared legit")
    print("=" * 78)
    ai_df = all_legit[(all_legit.dataset == "india") & (all_legit.model == "A")]
    fp = ai_df[ai_df.is_false_positive]
    ok = ai_df[~ai_df.is_false_positive]
    prof = pd.DataFrame(
        {
            "false_positive_mean": fp[PROFILE_FEATURES].mean().round(3),
            "cleared_mean": ok[PROFILE_FEATURES].mean().round(3),
        }
    )
    prof["ratio_fp/ok"] = (
        prof["false_positive_mean"] / prof["cleared_mean"].replace(0, np.nan)
    ).round(2)
    print(prof.to_string())

    # Bucketing of FP legit (India, Model A).
    print("\nFalse-positive legit by message type (India, Model A):")
    if len(fp):
        bk = fp["msg_type"].value_counts()
        for t, c in bk.items():
            print(f"  {t:22} {c}")

    # Examples with explanations (India, Model A FPs first; pad with W if few).
    print("\n" + "=" * 78)
    print("Example legit messages misclassified as 'unwanted' (with drivers)")
    print("=" * 78)
    examples = fp.sort_values("p_unwanted", ascending=False).head(10)
    if len(examples) < 8:
        extra = all_legit[
            (all_legit.dataset == "india")
            & (all_legit.model == "W")
            & all_legit.is_false_positive
        ]
        extra = extra.sort_values("p_unwanted", ascending=False).head(
            10 - len(examples)
        )
        examples = pd.concat([examples, extra], ignore_index=True)
    for _, r in examples.iterrows():
        pipe = A if r["model"] == "A" else W
        exp = explain(r["text"], pipe=pipe, k=6)
        drivers = ", ".join(
            f"{c['feature']}({c['contribution']:+.2f})"
            for c in exp["top_contributions"]
            if c["contribution"] > 0
        )
        print(
            f"\n  [{r['model']}|{r['dataset']}] P(unwanted)={r['p_unwanted']:.2f}  "
            f"type={r['msg_type']}  digits={r['digit_count']:.0f} "
            f"currency={r['count_currency']:.0f}"
        )
        print(f"    {r['text']}")
        print(f"    drivers: {drivers}")

    all_legit.to_csv(FIGURES_DIR / "false_positive_analysis.csv", index=False)
    print(
        f"\nSaved {len(all_legit)} legit-message rows to "
        f"{FIGURES_DIR / 'false_positive_analysis.csv'}"
    )
    print("Read-only: no models or data modified.")


if __name__ == "__main__":
    main()
