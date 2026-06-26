"""Phase 5b diagnostic — near-duplicate / template-similarity of the India split.

READ-ONLY. Does NOT retrain or modify the split, the models, or any Phase 5b
result. It loads the frozen ``india_train.csv`` / ``india_test.csv`` and the
existing Model A (``india_augmented_model.pkl``) to check whether Model A's
0.991 held-out India macro-F1 reflects genuine generalization or template
memorization (near-duplicates spanning the train/test boundary).

Two similarity measures per test message (max over the train set):
  (a) TF-IDF cosine on character n-grams (3-5) — catches templates with only
      numbers/dates changed.
  (b) Jaccard over the token set — complementary lexical-overlap view.

Run ``python -m src.diag_india_split``. Classical only; no neural nets.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

from src.config import PROJECT_ROOT
from src.eval_india import _ascii, build_eval_frame, to_unwanted_legit
from src.india_augment import INDIA_AUG_MODEL_PATH, INDIA_TEST_PATH, INDIA_TRAIN_PATH

FIGURES_DIR: Path = PROJECT_ROOT / "paper" / "figures"
NEARDUP_THRESHOLD: float = 0.90  # bucket cutoff (char-cosine)
SEED: int = 42

_TOKEN_RE = re.compile(r"\w+")
PROMO_KW = (
    "offer",
    "cashback",
    "recharge",
    "deal",
    " off",
    "sale",
    "plan",
    "data",
    "discount",
    "buy",
    "shop",
    "gb",
    "save",
    "bonus",
    "voucher",
    "coupon",
    "%",
)
SCAM_KW = (
    "kyc",
    "verify",
    "blocked",
    "otp",
    "suspend",
    "lottery",
    "won",
    "prize",
    "urgent",
    "update your",
    "account",
    "click here",
    "expire",
    "refund",
    "penalty",
    "fraud",
    "deactivat",
)
SENDER_KW = {
    "airtel": ("airtel",),
    "vi/vodafone": ("vi-", "vodafone", "vi india", "vi-india"),
    "jio": ("jio",),
    "bank/finance": ("bank", "sbi", "hdfc", "icici", "axis", "loan", "credit"),
    "ecommerce": ("amazon", "flipkart", "myntra", "smart point", "smart "),
}


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text).lower()))


def _jaccard_max(test_texts, train_texts) -> tuple[np.ndarray, np.ndarray]:
    """Max Jaccard of each test message to any train message (+ argmax index)."""
    train_sets = [_tokens(t) for t in train_texts]
    best_sim = np.zeros(len(test_texts))
    best_idx = np.zeros(len(test_texts), dtype=int)
    for i, t in enumerate(test_texts):
        a = _tokens(t)
        if not a:
            continue
        bi, bs = 0, 0.0
        for j, b in enumerate(train_sets):
            union = a | b
            if not union:
                continue
            s = len(a & b) / len(union)
            if s > bs:
                bs, bi = s, j
        best_sim[i], best_idx[i] = bs, bi
    return best_sim, best_idx


def _dist(name: str, vals: np.ndarray) -> dict:
    return {
        "measure": name,
        "min": float(vals.min()),
        "median": float(np.median(vals)),
        "mean": float(vals.mean()),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "n_ge_0.80": int((vals >= 0.80).sum()),
        "n_ge_0.90": int((vals >= 0.90).sum()),
        "n_ge_0.95": int((vals >= 0.95).sum()),
    }


def _collapse_unique(text_matrix, threshold: float) -> int:
    """Greedy near-duplicate collapse: count clusters at the given cosine sim."""
    sim = cosine_similarity(text_matrix)
    n = sim.shape[0]
    assigned = np.full(n, -1)
    reps: list[int] = []
    for i in range(n):
        placed = False
        for r in reps:
            if sim[i, r] >= threshold:
                assigned[i] = r
                placed = True
                break
        if not placed:
            reps.append(i)
            assigned[i] = i
    return len(reps)


def _classify_unwanted(text: str) -> str:
    t = str(text).lower()
    scam = any(k in t for k in SCAM_KW)
    promo = any(k in t for k in PROMO_KW)
    if scam and not promo:
        return "scam-like"
    if promo and not scam:
        return "promotional"
    if promo and scam:
        return "mixed"
    return "other"


def main() -> None:
    np.random.seed(SEED)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    india_train = pd.read_csv(INDIA_TRAIN_PATH)
    india_test = pd.read_csv(INDIA_TEST_PATH)
    train_text = india_train["text"].astype(str).tolist()
    test_text = india_test["text"].astype(str).tolist()

    print("=" * 78)
    print("Phase 5b diagnostic — India split near-duplicate / template similarity")
    print("=" * 78)
    print(
        f"india_train={len(train_text)}  india_test={len(test_text)}  "
        "(frozen; read-only)"
    )

    # Char n-gram TF-IDF fit on the union (similarity metric, not a model).
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    all_text = train_text + test_text
    M = vec.fit_transform(all_text)
    M_train = M[: len(train_text)]
    M_test = M[len(train_text) :]

    cos = cosine_similarity(M_test, M_train)  # 619 x 1442
    char_max = cos.max(axis=1)
    char_arg = cos.argmax(axis=1)

    jac_max, jac_arg = _jaccard_max(test_text, train_text)

    # ---- Similarity distributions ----
    char_dist = _dist("char_tfidf_cosine_3_5", char_max)
    jac_dist = _dist("token_jaccard", jac_max)
    print("\nMax similarity of each TEST message to the TRAIN set:")
    hdr = f"    {'measure':<26}{'min':>7}{'med':>7}{'mean':>7}{'p90':>7}{'p95':>7}{'p99':>7}{'>=.8':>7}{'>=.9':>7}{'>=.95':>7}"
    print(hdr)
    for d in (char_dist, jac_dist):
        print(
            f"    {d['measure']:<26}{d['min']:>7.3f}{d['median']:>7.3f}"
            f"{d['mean']:>7.3f}{d['p90']:>7.3f}{d['p95']:>7.3f}{d['p99']:>7.3f}"
            f"{d['n_ge_0.80']:>7}{d['n_ge_0.90']:>7}{d['n_ge_0.95']:>7}"
        )

    # ---- Bucketed Model A macro-F1 (the headline) ----
    model_A = joblib.load(INDIA_AUG_MODEL_PATH)
    y_true = to_unwanted_legit(india_test["label"])
    y_pred = model_A.predict(build_eval_frame(test_text))
    near_mask = char_max >= NEARDUP_THRESHOLD
    novel_mask = ~near_mask

    def _bucket_f1(mask) -> dict:
        if mask.sum() == 0:
            return {"n": 0, "macro_f1": None, "support": {}}
        yt, yp = y_true[mask], y_pred[mask]
        return {
            "n": int(mask.sum()),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "support": pd.Series(yt).value_counts().to_dict(),
        }

    near = _bucket_f1(near_mask)
    novel = _bucket_f1(novel_mask)
    overall = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    def _fmt(v) -> str:
        return "n/a" if v is None else f"{v:.4f}"

    print("\n" + "=" * 78)
    print(
        f"HEADLINE — Model A macro-F1 by bucket (near-dup cutoff char-cos "
        f">= {NEARDUP_THRESHOLD})"
    )
    print("=" * 78)
    print(f"    overall India test:            {overall:.4f}  (n={len(y_true)})")
    print(
        f"    near-duplicate bucket (>=0.9): {_fmt(near['macro_f1'])}"
        f"  (n={near['n']}, support={near['support']})"
    )
    print(
        f"    NOVEL bucket (<0.9):           {_fmt(novel['macro_f1'])}"
        f"  (n={novel['n']}, support={novel['support']})"
    )

    # ---- Lexical diversity (whole cleaned India set = train + test) ----
    M_all = M  # already fit on union (2061 messages)
    n_unique_090 = _collapse_unique(M_all, 0.90)
    full = pd.concat([india_train, india_test], ignore_index=True)
    unwanted = full[full["label"].str.lower() == "spam"].copy()
    sender_counts = {
        name: int(
            unwanted["text"]
            .str.lower()
            .apply(lambda t, kws=kws: any(k in t for k in kws))
            .sum()
        )
        for name, kws in SENDER_KW.items()
    }
    unwanted["kind"] = unwanted["text"].apply(_classify_unwanted)
    kind_frac = (unwanted["kind"].value_counts(normalize=True) * 100).round(1).to_dict()

    print("\n" + "=" * 78)
    print("Lexical diversity (full cleaned India set = 2061)")
    print("=" * 78)
    print(
        f"    unique messages after near-dup collapse @0.9: {n_unique_090} "
        f"/ {len(all_text)} ({100*n_unique_090/len(all_text):.1f}% unique)"
    )
    print(f"    unwanted ('spam') messages: {len(unwanted)}")
    print(f"    sender/template hits among unwanted: {sender_counts}")
    print(f"    unwanted composition (%): {kind_frac}")

    # ---- Example near-duplicate pairs (top by char cosine) ----
    order = np.argsort(-char_max)
    print("\n" + "=" * 78)
    print("Top test/train near-duplicate pairs (char-cosine)")
    print("=" * 78)
    examples = []
    for rank in order[:4]:
        tr_i = int(char_arg[rank])
        print(
            f"\n  sim={char_max[rank]:.3f}  jaccard={jac_max[rank]:.3f}  "
            f"(test label={india_test.iloc[rank]['label']})"
        )
        print(f"    TEST : {_ascii(test_text[rank], 150)}")
        print(f"    TRAIN: {_ascii(train_text[tr_i], 150)}")
        examples.append(
            {
                "char_cosine": round(float(char_max[rank]), 4),
                "jaccard": round(float(jac_max[rank]), 4),
                "test_label": india_test.iloc[rank]["label"],
                "test_text": _ascii(test_text[rank], 220),
                "nearest_train_text": _ascii(train_text[tr_i], 220),
            }
        )

    # ---- Persist ----
    per_msg = pd.DataFrame(
        {
            "test_text": [_ascii(t, 200) for t in test_text],
            "true_label": india_test["label"].values,
            "A_pred": y_pred,
            "char_cosine_max": np.round(char_max, 4),
            "jaccard_max": np.round(jac_max, 4),
            "bucket": np.where(near_mask, "near_dup_ge_0.9", "novel_lt_0.9"),
            "nearest_train_text": [_ascii(train_text[int(i)], 200) for i in char_arg],
        }
    )
    per_msg.to_csv(FIGURES_DIR / "india_split_similarity.csv", index=False)

    summary = {
        "seed": SEED,
        "neardup_threshold": NEARDUP_THRESHOLD,
        "n_train": len(train_text),
        "n_test": len(test_text),
        "distributions": {"char_tfidf_cosine": char_dist, "jaccard": jac_dist},
        "model_A_macro_f1": {
            "overall": overall,
            "near_duplicate": near,
            "novel": novel,
        },
        "diversity": {
            "n_total": len(all_text),
            "n_unique_collapse_0.9": n_unique_090,
            "pct_unique": round(100 * n_unique_090 / len(all_text), 1),
            "n_unwanted": int(len(unwanted)),
            "sender_hits_among_unwanted": sender_counts,
            "unwanted_composition_pct": kind_frac,
        },
        "example_pairs": examples,
    }
    with open(
        FIGURES_DIR / "india_split_similarity_summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {FIGURES_DIR / 'india_split_similarity.csv'}")
    print(f"       {FIGURES_DIR / 'india_split_similarity_summary.json'}")
    print("Read-only: split, models, and Phase 5b results untouched.")


if __name__ == "__main__":
    main()
