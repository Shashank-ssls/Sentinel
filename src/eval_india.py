"""Phase 5 — India-focused evaluation (hybrid, evaluation-only).

The production model (LogReg Stage-1, ``best_model.pkl``) is loaded and meets two
India sets exactly once. NOTHING here retrains or touches the Mendeley training
split; the Mendeley *test* split is only re-read to compute a comparable
generalization baseline.

Two India sets, two honest claims:

* **Dataset A — real Indian SMS** (``india_real_sms.csv``, binary ham/spam,
  real India-origin). Cleaned, then evaluated as a real-world **binary
  generalization** test: model spam|smishing -> "unwanted", ham -> "legit";
  dataset spam -> "unwanted", ham -> "legit". This is NOT a smishing-specific
  claim — the set has no smishing label and its "spam" is mostly promotional.
* **Dataset B — synthetic India-pattern probe** (``india_synthetic_sms.csv``,
  balanced 3-class). Evaluated full 3-class, focused on the smishing class the
  real set cannot test. This is a controlled pattern-level probe, NOT a
  real-world performance claim.

Run ``python -m src.eval_india``. Classical ML only; no neural nets.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.config import PROJECT_ROOT, get_config
from src.data import load_processed
from src.features import FEATURE_COLUMNS, derive_flags
from src.train import CLASS_ORDER

RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
REAL_PATH: Path = RAW_DIR / "india_real_sms.csv"
SYNTHETIC_PATH: Path = RAW_DIR / "india_synthetic_sms.csv"
FIGURES_DIR: Path = PROJECT_ROOT / "paper" / "figures"
PRODUCTION_RESULTS: Path = PROJECT_ROOT / "models" / "production_test_results.json"

BINARY_LABELS = ["legit", "unwanted"]

# Keyword buckets for real-set error analysis (Indian senders/scam types).
REAL_CATEGORIES: dict[str, tuple[str, ...]] = {
    "airtel": ("airtel",),
    "vi/vodafone": ("vi-", "vodafone", "vi india", "vi-india"),
    "jio": ("jio",),
    "govt/cert-in": ("cert-in", "csk.gov", "gov.in", " goi", "govt", "pakhwada"),
    "bank/kyc": ("bank", "sbi", "hdfc", "icici", "axis", "kyc", "account"),
    "ecommerce": ("amazon", "flipkart", "myntra", "order", "delivery"),
    "recharge/offer": ("recharge", "offer", "cashback", "data", "plan", "gb"),
}


def _ascii(text: str, limit: int = 110) -> str:
    """Console-safe truncation (avoids cp1252 crashes on Rs/currency glyphs)."""
    s = str(text).encode("ascii", "replace").decode("ascii")
    return s[:limit]


def load_model():
    """Load the production pipeline (LogReg Stage-1) from config.model_path."""
    path = get_config().model_path
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Production model not found at {path}. Run `python -m src.explain`."
        )
    return joblib.load(path)


def build_eval_frame(texts: list[str]) -> pd.DataFrame:
    """Build the model's input frame from raw texts (inference flag-derivation)."""
    rows = [{"text": str(t), **derive_flags(str(t))} for t in texts]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def to_unwanted_legit(labels) -> np.ndarray:
    """ham -> 'legit'; spam/smishing -> 'unwanted' (works for preds and truth)."""
    arr = np.asarray([str(x).strip().lower() for x in labels])
    return np.where(arr == "ham", "legit", "unwanted")


# --- Dataset A: real Indian SMS (binary) -----------------------------------


def clean_real(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the real set; return (clean_df[text,label], removal summary)."""
    df = df.rename(columns={"Msg": "text", "Label": "label"})
    start = len(df)

    text = df["text"].astype("string")
    null_mask = text.isna() | (text.str.strip() == "")
    n_null = int(null_mask.sum())
    df = df[~null_mask].copy()
    df["text"] = df["text"].astype(str)

    junk_mask = df["text"].str.contains("image omitted", case=False, na=False)
    n_junk = int(junk_mask.sum())
    df = df[~junk_mask]

    before_dedup = len(df)
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    n_dup = before_dedup - len(df)

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    summary = {
        "start_rows": start,
        "removed_null_or_empty": n_null,
        "removed_image_omitted": n_junk,
        "removed_duplicate_text": n_dup,
        "final_rows": len(df),
        "class_balance": df["label"].value_counts().to_dict(),
    }
    return df[["text", "label"]], summary


def evaluate_real(pipe, clean_df: pd.DataFrame) -> dict:
    """Binary (unwanted vs legit) evaluation on the real India set."""
    X = build_eval_frame(clean_df["text"].tolist())
    pred3 = pipe.predict(X)
    y_true = to_unwanted_legit(clean_df["label"])
    y_pred = to_unwanted_legit(pred3)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=BINARY_LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=BINARY_LABELS)
    return {
        "claim": "real-world binary generalization (unwanted vs legit)",
        "n": len(clean_df),
        "binary_macro_f1": macro_f1,
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": BINARY_LABELS,
        "_pred3": pred3,
        "_true_bin": y_true,
        "_pred_bin": y_pred,
    }


def _categorize_real(text: str) -> list[str]:
    t = text.lower()
    return [c for c, kws in REAL_CATEGORIES.items() if any(k in t for k in kws)]


def real_error_analysis(clean_df: pd.DataFrame, res: dict) -> dict:
    """Which Indian spam types are missed; false-alarm rate on ham."""
    df = clean_df.reset_index(drop=True).copy()
    df["true_bin"] = res["_true_bin"]
    df["pred_bin"] = res["_pred_bin"]
    df["pred3"] = res["_pred3"]

    missed = df[(df["true_bin"] == "unwanted") & (df["pred_bin"] == "legit")]
    false_alarm = df[(df["true_bin"] == "legit") & (df["pred_bin"] == "unwanted")]

    missed_cat: dict[str, int] = {c: 0 for c in REAL_CATEGORIES}
    missed_cat["uncategorized"] = 0
    for txt in missed["text"]:
        cats = _categorize_real(txt)
        if cats:
            for c in cats:
                missed_cat[c] += 1
        else:
            missed_cat["uncategorized"] += 1

    return {
        "n_unwanted": int((df["true_bin"] == "unwanted").sum()),
        "n_missed_unwanted": len(missed),
        "n_legit": int((df["true_bin"] == "legit").sum()),
        "n_false_alarm": len(false_alarm),
        "missed_by_category": missed_cat,
        "missed_examples": missed,
        "false_alarm_examples": false_alarm,
    }


# --- Dataset B: synthetic India-pattern probe (3-class) --------------------


def load_synthetic(path: Path = SYNTHETIC_PATH) -> pd.DataFrame:
    """Load synthetic probe, keeping text/label (+ notes for error analysis)."""
    df = pd.read_csv(path)
    keep = ["text", "label"] + (["notes"] if "notes" in df.columns else [])
    df = df[keep].copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df


def evaluate_synthetic(pipe, df: pd.DataFrame) -> dict:
    """Full 3-class evaluation on the synthetic India probe."""
    X = build_eval_frame(df["text"].tolist())
    pred = pipe.predict(X)
    y_true = df["label"].to_numpy()

    macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, pred, labels=list(CLASS_ORDER), output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, pred, labels=list(CLASS_ORDER))
    return {
        "claim": "controlled synthetic India-pattern 3-class probe",
        "n": len(df),
        "macro_f1": macro_f1,
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": list(CLASS_ORDER),
        "_pred": pred,
    }


def synthetic_error_analysis(df: pd.DataFrame, res: dict) -> dict:
    """Smishing recall by India scam category (from the notes column)."""
    df = df.reset_index(drop=True).copy()
    df["pred"] = res["_pred"]
    df["correct"] = df["pred"] == df["label"]

    by_cat = None
    if "notes" in df.columns:
        smish = df[df["label"] == "smishing"].copy()
        if len(smish):
            by_cat = (
                smish.groupby("notes")["correct"]
                .agg(n="count", caught="sum")
                .assign(recall=lambda d: d["caught"] / d["n"])
                .sort_values("recall")
                .reset_index()
            )
    misclassified = df[~df["correct"]]
    return {"smishing_recall_by_category": by_cat, "misclassified": misclassified}


# --- Generalization gap ----------------------------------------------------


def generalization_gap(pipe) -> pd.DataFrame:
    """Compare India results to comparable Mendeley held-out baselines.

    Computes the Mendeley test set's unwanted-vs-legit binary macro-F1 with the
    SAME mapping used for the real India set, so the binary comparison is fair.
    The 3-class baseline is read from the Phase 3 production results.
    """
    _, test_df = load_processed()
    pred = pipe.predict(test_df[FEATURE_COLUMNS])
    mendeley_binary = f1_score(
        to_unwanted_legit(test_df["label"]),
        to_unwanted_legit(pred),
        average="macro",
        zero_division=0,
    )
    mendeley_3class = json.loads(PRODUCTION_RESULTS.read_text())["macro_f1"]
    return pd.DataFrame(
        [
            {
                "evaluation": "Mendeley held-out (3-class)",
                "metric": "macro_f1_3class",
                "value": mendeley_3class,
            },
            {
                "evaluation": "Mendeley held-out (binary unwanted/legit)",
                "metric": "macro_f1_binary",
                "value": mendeley_binary,
            },
        ]
    )


# --- Orchestration ---------------------------------------------------------


def _save_confusion(cm, labels, path: Path) -> None:
    pd.DataFrame(
        cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels]
    ).to_csv(path)


def _print_report(res: dict, key: str, labels: list[str]) -> None:
    rep = res["per_class_report"]
    print(f"    {'class':<10}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
    for lbl in labels:
        r = rep[lbl]
        print(
            f"    {lbl:<10}{r['precision']:>8.3f}{r['recall']:>8.3f}"
            f"{r['f1-score']:>8.3f}{int(r['support']):>9}"
        )
    print(f"    {key}: {res[key]:.4f}")
    print("    confusion (rows=true, cols=pred): " + ", ".join(labels))
    for lbl, row in zip(labels, res["confusion_matrix"]):
        print(f"      {lbl:<10}" + "".join(f"{v:>8}" for v in row))


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pipe = load_model()
    examples = []

    print("=" * 74)
    print("Phase 5 — India evaluation (evaluation-only; training data untouched)")
    print("=" * 74)

    # ---- Dataset A: real Indian SMS (binary) ----
    real_raw = pd.read_csv(REAL_PATH)
    clean_df, summary = clean_real(real_raw)
    print("\n[A] Real Indian SMS — cleaning summary:")
    print(
        f"    start={summary['start_rows']}  "
        f"-null/empty={summary['removed_null_or_empty']}  "
        f"-image_omitted={summary['removed_image_omitted']}  "
        f"-dup_text={summary['removed_duplicate_text']}  "
        f"=> final={summary['final_rows']}"
    )
    print(f"    class balance: {summary['class_balance']}")

    real_res = evaluate_real(pipe, clean_df)
    print(
        "\n[A] Real set — BINARY (unwanted vs legit) "
        "[real-world generalization, NOT smishing-specific]:"
    )
    _print_report(real_res, "binary_macro_f1", BINARY_LABELS)
    real_err = real_error_analysis(clean_df, real_res)
    print(
        f"    missed unwanted: {real_err['n_missed_unwanted']}/"
        f"{real_err['n_unwanted']}  |  false alarms (ham->unwanted): "
        f"{real_err['n_false_alarm']}/{real_err['n_legit']}"
    )
    print(
        f"    missed unwanted by category: "
        f"{ {k: v for k, v in real_err['missed_by_category'].items() if v} }"
    )

    # ---- Dataset B: synthetic India probe (3-class) ----
    syn_df = load_synthetic()
    syn_res = evaluate_synthetic(pipe, syn_df)
    print(
        "\n[B] Synthetic India probe — 3-CLASS "
        "[controlled pattern probe, NOT a real-world claim]:"
    )
    _print_report(syn_res, "macro_f1", list(CLASS_ORDER))
    syn_err = synthetic_error_analysis(syn_df, syn_res)
    if syn_err["smishing_recall_by_category"] is not None:
        print("\n[B] Smishing recall by India scam category:")
        print(syn_err["smishing_recall_by_category"].to_string(index=False))

    # ---- Generalization gap ----
    gap = generalization_gap(pipe)
    print("\n" + "=" * 74)
    print("Generalization gap vs Mendeley held-out (comparable metrics)")
    print("=" * 74)
    m3 = gap.loc[gap.metric == "macro_f1_3class", "value"].iloc[0]
    mb = gap.loc[gap.metric == "macro_f1_binary", "value"].iloc[0]
    print(f"    Mendeley 3-class macro-F1:            {m3:.4f}")
    print(f"    Mendeley binary  macro-F1:            {mb:.4f}")
    print(
        f"    India REAL binary macro-F1:           {real_res['binary_macro_f1']:.4f}"
        f"   (gap {real_res['binary_macro_f1'] - mb:+.4f})"
    )
    print(
        f"    India SYNTH 3-class macro-F1:         {syn_res['macro_f1']:.4f}"
        f"   (gap {syn_res['macro_f1'] - m3:+.4f})"
    )

    # ---- Misclassification examples (both sets) ----
    for _, r in real_err["missed_examples"].head(4).iterrows():
        examples.append(
            {
                "dataset": "real",
                "true": "spam(unwanted)",
                "pred": f"ham(legit)/{r['pred3']}",
                "note": "missed unwanted",
                "text": _ascii(r["text"], 160),
            }
        )
    for _, r in real_err["false_alarm_examples"].head(2).iterrows():
        examples.append(
            {
                "dataset": "real",
                "true": "ham(legit)",
                "pred": f"unwanted/{r['pred3']}",
                "note": "false alarm",
                "text": _ascii(r["text"], 160),
            }
        )
    for _, r in syn_err["misclassified"].head(6).iterrows():
        examples.append(
            {
                "dataset": "synthetic",
                "true": r["label"],
                "pred": r["pred"],
                "note": r.get("notes", ""),
                "text": _ascii(r["text"], 160),
            }
        )
    ex_df = pd.DataFrame(examples)

    print("\n" + "=" * 74)
    print("Concrete misclassification examples")
    print("=" * 74)
    for e in examples[:6]:
        print(f"  [{e['dataset']}] true={e['true']} pred={e['pred']} ({e['note']})")
        print(f"      {e['text']}")

    # ---- Persist artifacts ----
    def _clean_res(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}

    with open(
        FIGURES_DIR / "india_real_binary_results.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "cleaning": summary,
                "results": _clean_res(real_res),
                "error_analysis": {
                    k: v for k, v in real_err.items() if not k.endswith("examples")
                },
            },
            f,
            indent=2,
        )
    _save_confusion(
        real_res["confusion_matrix"],
        BINARY_LABELS,
        FIGURES_DIR / "india_real_confusion.csv",
    )
    with open(
        FIGURES_DIR / "india_synthetic_3class_results.json", "w", encoding="utf-8"
    ) as f:
        json.dump(_clean_res(syn_res), f, indent=2)
    _save_confusion(
        syn_res["confusion_matrix"],
        list(CLASS_ORDER),
        FIGURES_DIR / "india_synthetic_confusion.csv",
    )
    gap.to_csv(FIGURES_DIR / "india_generalization_gap.csv", index=False)
    ex_df.to_csv(FIGURES_DIR / "india_misclassification_examples.csv", index=False)
    if syn_err["smishing_recall_by_category"] is not None:
        syn_err["smishing_recall_by_category"].to_csv(
            FIGURES_DIR / "india_synthetic_smishing_recall_by_category.csv", index=False
        )

    print(f"\nArtifacts written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
