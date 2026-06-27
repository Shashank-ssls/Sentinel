"""Read-only diagnostic — where Model W and Model A disagree on India test.

Finds curated demo candidates: messages the Western-only model (W) clears but
the India-augmented model (A) flags, with A correct. Uses the existing models
and the frozen ``india_test.csv``; trains/changes NOTHING.

For each test message: W/A prediction + confidence + P(unwanted). Disagreements
are categorized and, for the clean "W-miss / A-catch / A-correct" cases, the
India-specific terms A weighted (that W could not) are surfaced via explain().

Run ``python -m src.diag_wa_disagreements``. Output: console summary +
``paper/figures/wa_disagreements.csv``.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.eval_india import _ascii, build_eval_frame, to_unwanted_legit
from src.explain import explain, get_coefficients
from src.india_augment import INDIA_AUG_MODEL_PATH, INDIA_TEST_PATH
from src.train import MODELS_DIR

WESTERN_PATH: Path = MODELS_DIR / "western_binary_model.pkl"
FIGURES_DIR: Path = PROJECT_ROOT / "paper" / "figures"
CONF_GAP = 0.30  # "meaningful" confidence divergence threshold

# India-specific cues we expect A to learn and W to lack.
INDIA_CUES = (
    "airtel",
    "jio",
    "vodafone",
    "vi ",
    ".in",
    "upi",
    "icici",
    "sbi",
    "hdfc",
    "axis",
    "kotak",
    "kyc",
    "aadhaar",
    "fastag",
    "paytm",
    "phonepe",
    "yono",
    "rs",
    "inr",
    "recharge",
    "rupees",
    "lakh",
    "crore",
)


def _unwanted_weight_map(pipe) -> dict[str, float]:
    """Map raw feature name -> coefficient toward the 'unwanted' class."""
    classes, names, coef = get_coefficients(pipe)
    u = classes.index("unwanted")
    return {n: float(coef[u, i]) for i, n in enumerate(names)}


def _is_india_term(label: str) -> bool:
    t = label.lower()
    return any(cue.strip() in t for cue in INDIA_CUES)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    W = joblib.load(WESTERN_PATH)
    A = joblib.load(INDIA_AUG_MODEL_PATH)
    test = pd.read_csv(INDIA_TEST_PATH)
    texts = test["text"].astype(str).tolist()
    true_bin = to_unwanted_legit(test["label"])

    X = build_eval_frame(texts)
    classes = list(W.classes_)  # ['legit', 'unwanted'] for both
    u_idx = classes.index("unwanted")

    pw, pa = W.predict_proba(X), A.predict_proba(X)
    W_pred = np.array([classes[i] for i in pw.argmax(1)])
    A_pred = np.array([classes[i] for i in pa.argmax(1)])
    W_conf, A_conf = pw.max(1), pa.max(1)
    W_pu, A_pu = pw[:, u_idx], pa[:, u_idx]

    def categorize(i: int) -> str:
        if W_pred[i] != A_pred[i]:
            if W_pred[i] == "legit" and A_pred[i] == "unwanted":
                return (
                    "miss_catch_correct"
                    if true_bin[i] == "unwanted"
                    else "miss_catch_wrong"
                )
            return "W_flag_A_clear"  # W=unwanted, A=legit
        if abs(A_conf[i] - W_conf[i]) >= CONF_GAP:
            return "agree_conf_gap"
        return "agree"

    df = pd.DataFrame(
        {
            "text": texts,
            "true_label": true_bin,
            "W_pred": W_pred,
            "W_conf": np.round(W_conf, 4),
            "W_p_unwanted": np.round(W_pu, 4),
            "A_pred": A_pred,
            "A_conf": np.round(A_conf, 4),
            "A_p_unwanted": np.round(A_pu, 4),
            "p_unwanted_gap": np.round(A_pu - W_pu, 4),
            "category": [categorize(i) for i in range(len(texts))],
        }
    )

    n = len(df)
    label_disagree = (df["W_pred"] != df["A_pred"]).sum()
    clean = df[df["category"] == "miss_catch_correct"].copy()
    miss_catch_wrong = (df["category"] == "miss_catch_wrong").sum()
    w_flag_a_clear = (df["category"] == "W_flag_A_clear").sum()
    conf_gap = (df["category"] == "agree_conf_gap").sum()
    meaningful = (
        (df["W_pred"] != df["A_pred"]) | (df["p_unwanted_gap"].abs() >= CONF_GAP)
    ).sum()

    print("=" * 78)
    print("W vs A disagreement diagnostic — India held-out test (read-only)")
    print("=" * 78)
    print(f"test messages:                      {n}")
    print(
        f"label disagreements (total):        {label_disagree} "
        f"({100*label_disagree/n:.1f}%)"
    )
    print(f"  clean W-miss / A-catch / CORRECT: {len(clean)}")
    print(f"  W-miss / A-catch but WRONG:       {miss_catch_wrong}")
    print(f"  W-flag / A-clear (A worse?):      {w_flag_a_clear}")
    print(f"agree-on-label but conf gap >= {CONF_GAP}: {conf_gap}")
    print(
        f"meaningful per-message divergence:  {meaningful} "
        f"({100*meaningful/n:.1f}%)  [label flip OR |P(unwanted) gap| >= {CONF_GAP}]"
    )

    # Honesty line on the W-flag/A-clear cases.
    if w_flag_a_clear:
        wfa = df[df["category"] == "W_flag_A_clear"]
        wfa_wrong = (wfa["true_label"] == "unwanted").sum()
        print(
            f"  (of W-flag/A-clear: {wfa_wrong} were truly unwanted -> A newly WRONG)"
        )

    # ---- Top clean cases, ranked by A_conf - W_conf ----
    clean["conf_margin"] = (clean["A_conf"] - clean["W_conf"]).round(4)
    clean = clean.sort_values("conf_margin", ascending=False).reset_index(drop=True)

    wmap, amap = _unwanted_weight_map(W), _unwanted_weight_map(A)

    print("\n" + "=" * 78)
    print(
        f"TOP CLEAN DISAGREEMENTS (W=legit, A=unwanted, true=unwanted): "
        f"{len(clean)} total"
    )
    print("=" * 78)
    india_terms_col = []
    for _, r in clean.head(10).iterrows():
        expA = explain(r["text"], pipe=A, k=10)
        india_hits = [
            c
            for c in expA["top_contributions"]
            if c["contribution"] > 0 and _is_india_term(c["feature"])
        ]
        terms_str = "; ".join(
            f"{c['feature']}({c['contribution']:+.2f})" for c in india_hits
        )
        print(
            f"\n  margin={r['conf_margin']:+.2f}  "
            f"W=legit({r['W_conf']:.2f})  A=unwanted({r['A_conf']:.2f})"
        )
        print(f"    {_ascii(r['text'], 150)}")
        # India terms A weighted, and what W did with the same term.
        if india_hits:
            print("    India terms A learned (A weight -> unwanted | W weight):")
            for c in india_hits[:5]:
                name = ("tfidf__" if c["kind"] == "term" else c["kind"] + "__") + c[
                    "feature"
                ]
                ww = wmap.get(name)
                wtxt = "absent in W" if ww is None else f"{ww:+.2f}"
                print(f"      {c['feature']:<22} A {c['contribution']:+.2f}   W {wtxt}")
        else:
            top = ", ".join(
                f"{c['feature']}({c['contribution']:+.2f})"
                for c in expA["top_contributions"][:4]
                if c["contribution"] > 0
            )
            print(f"    (A's drivers: {top})")
        india_terms_col.append(terms_str)

    # ---- Persist (disagreements + conf-gap agreements) ----
    keep = df[
        df["category"].isin(
            [
                "miss_catch_correct",
                "miss_catch_wrong",
                "W_flag_A_clear",
                "agree_conf_gap",
            ]
        )
    ].copy()
    keep["conf_margin"] = (keep["A_conf"] - keep["W_conf"]).round(4)
    keep["text"] = keep["text"].apply(lambda t: _ascii(t, 240))
    keep = keep.sort_values(["category", "conf_margin"], ascending=[True, False])
    keep.to_csv(FIGURES_DIR / "wa_disagreements.csv", index=False)
    print(f"\nSaved {len(keep)} rows to {FIGURES_DIR / 'wa_disagreements.csv'}")
    print("Read-only: no models or data modified.")


if __name__ == "__main__":
    main()
