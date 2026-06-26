"""Phase 5b — India-augmented model (the fix).

Phase 5 quantified the India generalization gap. Here we show it can be closed
by training with India data. Binary only (unwanted vs legit), since the real
India set is binary.

Two models, identical LogReg/pipeline/seed, evaluated on the SAME held-out sets:

* **Model W (Western-only):** trained on the Mendeley training split (binary).
  The baseline.
* **Model A (India-augmented):** trained on Mendeley training + the 70% India
  train split (binary).

The 30% India test split is sacred — created here with a fixed seed, saved to
``data/processed/``, and never used in training or TF-IDF fitting. Both the
Western-only model and all Phase 5 results stay intact (we report both).

Binary mapping (everywhere this phase): Mendeley spam+smishing -> "unwanted",
ham -> "legit"; India spam -> "unwanted", ham -> "legit".

Run ``python -m src.india_augment``. Classical ML only; no neural nets.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.config import PROJECT_ROOT, get_config
from src.data import load_processed
from src.eval_india import (
    BINARY_LABELS,
    REAL_PATH,
    _ascii,
    build_eval_frame,
    clean_real,
    to_unwanted_legit,
)
from src.explain import explain
from src.features import FEATURE_COLUMNS
from src.train import MODELS_DIR, logreg_pipeline

PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
INDIA_TRAIN_PATH: Path = PROCESSED_DIR / "india_train.csv"
INDIA_TEST_PATH: Path = PROCESSED_DIR / "india_test.csv"
FIGURES_DIR: Path = PROJECT_ROOT / "paper" / "figures"

WESTERN_MODEL_PATH: Path = MODELS_DIR / "western_binary_model.pkl"
INDIA_AUG_MODEL_PATH: Path = MODELS_DIR / "india_augmented_model.pkl"

INDIA_TEST_SIZE: float = 0.30
N_SPLITS: int = 5
INDIA_TERMS = ("upi", "kyc", "aadhaar", "fastag", "rs", "inr", ".in", "yono", "paytm")


# --- India split (sacred 30% held out) -------------------------------------


def make_india_split(
    clean_df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split of the cleaned real India set (pure)."""
    train_df, test_df = train_test_split(
        clean_df,
        test_size=test_size,
        random_state=seed,
        stratify=clean_df["label"],
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_india_split(
    seed: int | None = None, test_size: float = INDIA_TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the real India set, split 70/30, and persist to data/processed/."""
    seed = get_config().random_seed if seed is None else seed
    clean_df, _ = clean_real(pd.read_csv(REAL_PATH))
    india_train, india_test = make_india_split(clean_df, test_size, seed)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    india_train.to_csv(INDIA_TRAIN_PATH, index=False)
    india_test.to_csv(INDIA_TEST_PATH, index=False)
    return india_train, india_test


# --- Train / evaluate ------------------------------------------------------


def train_binary_model(X: pd.DataFrame, y, seed: int):
    """Fit a binary LogReg pipeline (same pipeline as the rest of the project)."""
    pipe = logreg_pipeline(seed, include_url=False)
    pipe.fit(X, y)
    return pipe


def evaluate_binary(pipe, X: pd.DataFrame, y_true) -> dict:
    """Binary (unwanted vs legit) metrics for a fitted pipeline."""
    y_pred = pipe.predict(X)
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class_report": classification_report(
            y_true, y_pred, labels=BINARY_LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=BINARY_LABELS
        ).tolist(),
        "_pred": y_pred,
    }


def cv_macro_f1(X: pd.DataFrame, y, seed: int) -> tuple[float, float]:
    """5-fold stratified CV macro-F1 (mean, std) on a model's training set."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    scores = cross_val_score(
        logreg_pipeline(seed, include_url=False), X, y, cv=cv, scoring="f1_macro"
    )
    return float(scores.mean()), float(scores.std())


# --- Orchestration ---------------------------------------------------------


def _report_line(name: str, res: dict) -> None:
    rep = res["per_class_report"]
    print(f"  {name}  macro-F1={res['macro_f1']:.4f}")
    print(f"    {'class':<10}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
    for lbl in BINARY_LABELS:
        r = rep[lbl]
        print(
            f"    {lbl:<10}{r['precision']:>8.3f}{r['recall']:>8.3f}"
            f"{r['f1-score']:>8.3f}{int(r['support']):>9}"
        )
    cm = res["confusion_matrix"]
    print("    confusion (rows=true, cols=pred): " + ", ".join(BINARY_LABELS))
    for lbl, row in zip(BINARY_LABELS, cm):
        print(f"      {lbl:<10}" + "".join(f"{v:>8}" for v in row))


def main() -> None:
    seed = get_config().random_seed
    np.random.seed(seed)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Data: Mendeley (schema flags) + India (derived flags), all binary.
    mend_train, mend_test = load_processed()
    india_train, india_test = build_india_split(seed)

    X_mend_tr = mend_train[FEATURE_COLUMNS].reset_index(drop=True)
    y_mend_tr = to_unwanted_legit(mend_train["label"])
    X_india_tr = build_eval_frame(india_train["text"].tolist())
    y_india_tr = to_unwanted_legit(india_train["label"])

    X_train_A = pd.concat([X_mend_tr, X_india_tr], ignore_index=True)
    y_train_A = np.concatenate([y_mend_tr, y_india_tr])

    X_test_mend = mend_test[FEATURE_COLUMNS].reset_index(drop=True)
    y_test_mend = to_unwanted_legit(mend_test["label"])
    X_test_india = build_eval_frame(india_test["text"].tolist())
    y_test_india = to_unwanted_legit(india_test["label"])

    print("=" * 76)
    print("Phase 5b — India-augmented model (binary unwanted vs legit)")
    print("=" * 76)
    print(
        f"Mendeley train={len(X_mend_tr)}  India train={len(X_india_tr)} "
        f"(70%)  India test={len(X_test_india)} (30%, sacred)  "
        f"Mendeley test={len(X_test_mend)}"
    )

    # Train both models.
    model_W = train_binary_model(X_mend_tr, y_mend_tr, seed)
    model_A = train_binary_model(X_train_A, y_train_A, seed)
    joblib.dump(model_W, WESTERN_MODEL_PATH)
    joblib.dump(model_A, INDIA_AUG_MODEL_PATH)

    # CV on each training set (error bars).
    cvW = cv_macro_f1(X_mend_tr, y_mend_tr, seed)
    cvA = cv_macro_f1(X_train_A, y_train_A, seed)

    # Evaluate both models on both test sets.
    results = {
        ("W", "Mendeley"): evaluate_binary(model_W, X_test_mend, y_test_mend),
        ("W", "India"): evaluate_binary(model_W, X_test_india, y_test_india),
        ("A", "Mendeley"): evaluate_binary(model_A, X_test_mend, y_test_mend),
        ("A", "India"): evaluate_binary(model_A, X_test_india, y_test_india),
    }
    cv_by_model = {"W": cvW, "A": cvA}
    names = {
        "W": "Model W (Western-only)",
        "A": "Model A (India-augmented)",
    }

    # Comparison table (2 models x 2 test sets).
    rows = []
    for (m, ts), res in results.items():
        rep = res["per_class_report"]
        rows.append(
            {
                "model": names[m],
                "test_set": ts,
                "macro_f1": res["macro_f1"],
                "f1_legit": rep["legit"]["f1-score"],
                "f1_unwanted": rep["unwanted"]["f1-score"],
                "cv_macro_f1_mean": cv_by_model[m][0],
                "cv_macro_f1_std": cv_by_model[m][1],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(FIGURES_DIR / "india_augment_comparison.csv", index=False)

    print("\nComparison table (2 models x 2 test sets):")
    print(table.round(4).to_string(index=False))

    print("\nCV macro-F1 on training set (error bars):")
    print(f"    Model W: {cvW[0]:.4f} +/- {cvW[1]:.4f}")
    print(f"    Model A: {cvA[0]:.4f} +/- {cvA[1]:.4f}")

    print("\nPer-model / per-test detail:")
    for m in ("W", "A"):
        print(f"\n{names[m]}:")
        for ts in ("Mendeley", "India"):
            _report_line(f"on {ts} test", results[(m, ts)])

    # Headline deltas.
    india_gain = (
        results[("A", "India")]["macro_f1"] - results[("W", "India")]["macro_f1"]
    )
    western_delta = (
        results[("A", "Mendeley")]["macro_f1"] - results[("W", "Mendeley")]["macro_f1"]
    )
    print("\n" + "=" * 76)
    print("HEADLINE")
    print("=" * 76)
    print(
        f"India held-out macro-F1: W={results[('W','India')]['macro_f1']:.4f} -> "
        f"A={results[('A','India')]['macro_f1']:.4f}   (gain {india_gain:+.4f})"
    )
    print(
        f"Mendeley held-out macro-F1: W={results[('W','Mendeley')]['macro_f1']:.4f} -> "
        f"A={results[('A','Mendeley')]['macro_f1']:.4f}   "
        f"(change {western_delta:+.4f} -> "
        f"{'held steady' if abs(western_delta) < 0.01 else 'REGRESSED' if western_delta < 0 else 'improved'})"
    )

    # Newly-caught India examples: true unwanted, W missed (legit), A caught.
    predW = results[("W", "India")]["_pred"]
    predA = results[("A", "India")]["_pred"]
    caught_mask = (
        (y_test_india == "unwanted") & (predW == "legit") & (predA == "unwanted")
    )
    caught = india_test[caught_mask].reset_index(drop=True)

    print("\n" + "=" * 76)
    print(f"Newly-caught India messages (W missed -> A caught): {len(caught)} total")
    print("=" * 76)
    newly = []
    for _, r in caught.head(4).iterrows():
        exp = explain(r["text"], pipe=model_A, k=8)
        india_hits = [
            c
            for c in exp["top_contributions"]
            if any(t in c["feature"].lower() for t in INDIA_TERMS)
        ]
        top_terms = ", ".join(
            f"{c['feature']}({c['contribution']:+.2f})"
            for c in exp["top_contributions"][:5]
        )
        print(f"\n  {_ascii(r['text'], 150)}")
        print(
            f"    A predicts: {exp['predicted']} "
            f"(unwanted p={exp['probabilities'].get('unwanted', 0):.3f})"
        )
        print(f"    top A contributions: {top_terms}")
        if india_hits:
            print(
                "    India-specific terms learned: "
                + ", ".join(
                    f"{c['feature']}({c['contribution']:+.2f})" for c in india_hits
                )
            )
        newly.append(
            {
                "text": _ascii(r["text"], 200),
                "A_predicted": exp["predicted"],
                "A_top_terms": top_terms,
                "india_terms": "; ".join(c["feature"] for c in india_hits),
            }
        )
    pd.DataFrame(newly).to_csv(
        FIGURES_DIR / "india_augment_newly_caught.csv", index=False
    )

    # Persist full results JSON.
    payload = {
        "seed": seed,
        "sizes": {
            "mendeley_train": len(X_mend_tr),
            "india_train": len(X_india_tr),
            "india_test": len(X_test_india),
            "mendeley_test": len(X_test_mend),
        },
        "cv": {
            "W": {"mean": cvW[0], "std": cvW[1]},
            "A": {"mean": cvA[0], "std": cvA[1]},
        },
        "results": {
            f"{m}_on_{ts}": {k: v for k, v in res.items() if not k.startswith("_")}
            for (m, ts), res in results.items()
        },
        "headline": {
            "india_gain": india_gain,
            "western_delta": western_delta,
            "n_newly_caught": int(caught_mask.sum()),
        },
    }
    with open(FIGURES_DIR / "india_augment_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved: {WESTERN_MODEL_PATH.name}, {INDIA_AUG_MODEL_PATH.name}")
    print(f"Artifacts in {FIGURES_DIR}")
    print("Phase 5 results and the 3-class production model are untouched.")


if __name__ == "__main__":
    main()
