"""Export the India-augmented binary model (Model A) to a sklearn-free JSON.

Part A of the Sentinel Android-app pivot. This script is **read-only** with
respect to the trained model: it loads ``models/india_augmented_model.pkl``,
serializes everything needed to reproduce a prediction from raw message text,
and writes a single ``export/sentinel_model.json``. scikit-learn is used ONLY
to load the pickle and to cross-check parity — never to consume the JSON.

The pipeline being exported (all classical, no neural nets, no network):

    Pipeline(
        features=Pipeline(
            columns=ColumnTransformer(tfidf=TfidfVectorizer, struct=StructuralFeatures),
            scale=MaxAbsScaler,
        ),
        clf=LogisticRegression(binary),
    )

Inference order the JSON consumer (e.g. the Kotlin port) must replicate:
    1. lowercase text, tokenize with ``token_pattern``;
    2. build 1- and 2-grams (bigrams joined by a single space);
    3. tf = 1 + ln(count)  (sublinear_tf); value = tf * idf[index];
    4. L2-normalize the TF-IDF row;
    5. compute the 10 structural features (regex + arithmetic);
    6. divide EVERY feature (tfidf block + struct block) by ``max_abs[i]``;
    7. dot with ``coef`` (+ intercept), sigmoid, threshold 0.5 -> "unwanted".

Run: ``python -m export.export_model`` (or ``python export/export_model.py``).
Writes the JSON only if the built-in 20-message parity check passes.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Make ``src`` importable so joblib can unpickle the custom transformers.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import message_to_frame  # noqa: E402  (after sys.path tweak)

FORMAT_VERSION = "1.0"
MODEL_FILE = "india_augmented_model.pkl"
MODEL_PATH = PROJECT_ROOT / "models" / MODEL_FILE
INDIA_REAL_CSV = PROJECT_ROOT / "data" / "raw" / "india_real_sms.csv"
OUT_PATH = Path(__file__).resolve().parent / "sentinel_model.json"

PARITY_N = 20
PARITY_SEED = 42
PARITY_TOL = 1e-6

# Currency symbols and urgency keywords mirror src/features.py exactly. They are
# embedded in the JSON so the port has a single source of truth.
CURRENCY_CHARSET = "$£€₹¥"
URGENCY_KEYWORDS = ["verify", "blocked", "kyc", "win", "urgent", "otp", "free", "prize"]
URGENCY_REGEX = r"\b(" + "|".join(re.escape(k) for k in URGENCY_KEYWORDS) + r")\b"
URL_REGEX = r"(https?://|www\.)"
PHONE_REGEX = r"(?:\+?\d[\s-]?){7,}"
EMAIL_REGEX = r"[\w.+-]+@[\w-]+\.[\w.-]+"


def _git_commit() -> str:
    """Best-effort HEAD commit hash of this repo (empty string if unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git absent / not a repo
        return ""


# --------------------------------------------------------------------------
# Build the export payload from the fitted pipeline.
# --------------------------------------------------------------------------


def build_payload(model) -> dict:
    """Extract a fully self-contained, sklearn-free spec from the pipeline."""
    feats = model.named_steps["features"]
    ct = feats.named_steps["columns"]
    scaler = feats.named_steps["scale"]
    clf = model.named_steps["clf"]

    named = dict((n, t) for n, t, _ in ct.transformers_)
    tfidf = named["tfidf"]
    struct = named["struct"]

    struct_names = [str(s) for s in struct.get_feature_names_out()]
    vocab_size = len(tfidf.vocabulary_)
    n_features = int(clf.coef_.shape[1])
    assert n_features == vocab_size + len(struct_names), "feature count mismatch"
    assert int(scaler.max_abs_.shape[0]) == n_features, "scaler width mismatch"

    classes = [str(c) for c in clf.classes_]  # ['legit', 'unwanted']

    payload = {
        "format_version": FORMAT_VERSION,
        "model_file": MODEL_FILE,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "vectorizer": {
            "analyzer": tfidf.analyzer,
            "lowercase": bool(tfidf.lowercase),
            "token_pattern": tfidf.token_pattern,
            "ngram_range": list(tfidf.ngram_range),
            "norm": tfidf.norm,
            "use_idf": bool(tfidf.use_idf),
            "smooth_idf": bool(tfidf.smooth_idf),
            "sublinear_tf": bool(tfidf.sublinear_tf),
            "binary": bool(tfidf.binary),
            "min_df": tfidf.min_df,
            "stop_words": tfidf.stop_words,
            "strip_accents": tfidf.strip_accents,
            "vocabulary": {str(k): int(v) for k, v in tfidf.vocabulary_.items()},
            "idf": [float(x) for x in tfidf.idf_],
        },
        "structural_features": {
            "names": struct_names,
            "specs": [
                {"name": "message_length", "kind": "int",
                 "spec": "number of Unicode code points in the raw text (len(text))"},
                {"name": "word_count", "kind": "int",
                 "spec": "count of non-empty tokens after splitting on Unicode whitespace runs (text.split())"},
                {"name": "digit_count", "kind": "int",
                 "spec": "count of characters c where c.isdigit() (Unicode-aware)"},
                {"name": "uppercase_ratio", "kind": "float",
                 "spec": "uppercase_alpha / total_alpha over chars where isalpha(); 0.0 if no alpha chars"},
                {"name": "count_links", "kind": "int",
                 "spec": "number of non-overlapping matches of url_regex (case-insensitive)"},
                {"name": "has_url", "kind": "bool01",
                 "spec": "1.0 if url_regex found anywhere (case-insensitive) else 0.0"},
                {"name": "has_phone", "kind": "bool01",
                 "spec": "1.0 if phone_regex matches anywhere else 0.0"},
                {"name": "has_email", "kind": "bool01",
                 "spec": "1.0 if email_regex matches anywhere else 0.0"},
                {"name": "count_currency", "kind": "int",
                 "spec": "total occurrences of any character in currency_charset"},
                {"name": "urgency_count", "kind": "int",
                 "spec": "number of non-overlapping matches of urgency_regex (case-insensitive, word-bounded)"},
            ],
            "regexes": {
                "url_regex": URL_REGEX,
                "phone_regex": PHONE_REGEX,
                "email_regex": EMAIL_REGEX,
                "urgency_regex": URGENCY_REGEX,
            },
            "currency_charset": CURRENCY_CHARSET,
            "urgency_keywords": URGENCY_KEYWORDS,
            "note": (
                "has_url/has_phone/has_email are DERIVED from the raw text via the "
                "regexes above (same code path as training-time inference). The port "
                "must derive them from text, not receive them as input."
            ),
        },
        "scaler": {
            "type": "max_abs",
            "rule": "divide each feature i by max_abs[i] (index in global feature_order)",
            "max_abs": [float(x) for x in scaler.max_abs_],
        },
        "classifier": {
            "type": "LogisticRegression",
            "classes": classes,
            "positive_class": classes[1],
            "positive_class_index": 1,
            "coef": [float(x) for x in clf.coef_[0]],
            "intercept": float(clf.intercept_[0]),
            "decision": "p_positive = sigmoid(dot(x, coef) + intercept)",
            "threshold": 0.5,
            "predict_rule": "predict classes[1] iff p_positive >= threshold else classes[0]",
        },
        "feature_order": {
            "layout": "[ tfidf block (vocabulary index order), structural block (names order) ]",
            "n_features": n_features,
            "blocks": [
                {
                    "name": "tfidf",
                    "size": vocab_size,
                    "global_offset": 0,
                    "index_source": "vectorizer.vocabulary maps token -> column index",
                },
                {
                    "name": "structural",
                    "size": len(struct_names),
                    "global_offset": vocab_size,
                    "names": struct_names,
                },
            ],
        },
    }
    return payload


# --------------------------------------------------------------------------
# Pure-Python reference inference — reads ONLY the payload, no sklearn.
# This doubles as the spec the Kotlin port must replicate byte-for-byte.
# --------------------------------------------------------------------------


def _structural_vector(text: str, m: dict) -> list[float]:
    """Compute the 10 structural features from raw text, JSON-driven."""
    text = "" if text is None else str(text)
    rx = m["structural_features"]["regexes"]
    url_re = re.compile(rx["url_regex"], re.IGNORECASE)
    phone_re = re.compile(rx["phone_regex"])
    email_re = re.compile(rx["email_regex"])
    urgency_re = re.compile(rx["urgency_regex"], re.IGNORECASE)
    charset = m["structural_features"]["currency_charset"]

    length = float(len(text))
    word_count = float(len(text.split()))
    digit_count = float(sum(ch.isdigit() for ch in text))
    alpha = [ch for ch in text if ch.isalpha()]
    uppercase_ratio = (
        sum(ch.isupper() for ch in alpha) / len(alpha) if alpha else 0.0
    )
    count_links = float(len(url_re.findall(text)))
    has_url = float(bool(url_re.search(text)))
    has_phone = float(bool(phone_re.search(text)))
    has_email = float(bool(email_re.search(text)))
    count_currency = float(sum(text.count(sym) for sym in charset))
    urgency_count = float(len(urgency_re.findall(text)))
    return [
        length, word_count, digit_count, uppercase_ratio, count_links,
        has_url, has_phone, has_email, count_currency, urgency_count,
    ]


def reference_predict(text: str, m: dict) -> tuple[str, float]:
    """Predict (label, p_positive) from raw text using ONLY the JSON payload."""
    text = "" if text is None else str(text)
    vec = m["vectorizer"]
    vocab = vec["vocabulary"]
    idf = vec["idf"]
    token_re = re.compile(vec["token_pattern"])

    tokens = token_re.findall(text.lower() if vec["lowercase"] else text)
    lo, hi = vec["ngram_range"]
    grams: list[str] = []
    for n in range(lo, hi + 1):
        if n == 1:
            grams.extend(tokens)
        else:
            grams.extend(
                " ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
            )

    counts: dict[int, int] = {}
    for g in grams:
        idx = vocab.get(g)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1

    # tf-idf with sublinear tf, then L2-normalize the row.
    vals: dict[int, float] = {}
    for idx, c in counts.items():
        tf = 1.0 + math.log(c) if vec["sublinear_tf"] else float(c)
        vals[idx] = tf * idf[idx]
    if vec["norm"] == "l2":
        norm = math.sqrt(sum(v * v for v in vals.values()))
        if norm > 0.0:
            for idx in vals:
                vals[idx] /= norm

    max_abs = m["scaler"]["max_abs"]
    coef = m["classifier"]["coef"]
    offset = m["feature_order"]["blocks"][1]["global_offset"]

    dot = m["classifier"]["intercept"]
    for idx, v in vals.items():
        dot += coef[idx] * (v / max_abs[idx])
    struct = _structural_vector(text, m)
    for j, sv in enumerate(struct):
        gi = offset + j
        dot += coef[gi] * (sv / max_abs[gi])

    p_pos = 1.0 / (1.0 + math.exp(-dot))
    classes = m["classifier"]["classes"]
    label = classes[1] if p_pos >= m["classifier"]["threshold"] else classes[0]
    return label, p_pos


# --------------------------------------------------------------------------
# Parity check + main.
# --------------------------------------------------------------------------


def run_parity(model, payload: dict) -> None:
    """Assert sklearn and the JSON reference agree on 20 sampled messages."""
    df = pd.read_csv(INDIA_REAL_CSV)
    sample = df.sample(n=PARITY_N, random_state=PARITY_SEED).reset_index(drop=True)
    pos_idx = list(model.named_steps["clf"].classes_).index(
        payload["classifier"]["positive_class"]
    )

    print(f"\nParity check ({PARITY_N} messages from {INDIA_REAL_CSV.name}, "
          f"seed={PARITY_SEED}, tol={PARITY_TOL:g}):")
    print(f"  {'#':>2}  {'sklearn':<9}{'ref':<9}{'p_sklearn':>11}{'p_ref':>11}"
          f"{'|diff|':>12}  match  text")
    failures = 0
    for i, text in enumerate(sample["Msg"].tolist()):
        frame = message_to_frame(text)
        sk_label = str(model.predict(frame)[0])
        sk_p = float(model.predict_proba(frame)[0][pos_idx])
        ref_label, ref_p = reference_predict(text, payload)
        diff = abs(sk_p - ref_p)
        ok = (sk_label == ref_label) and (diff < PARITY_TOL)
        if not ok:
            failures += 1
        snippet = text.replace("\n", " ")[:32].encode("ascii", "replace").decode("ascii")
        print(f"  {i:>2}  {sk_label:<9}{ref_label:<9}{sk_p:>11.6f}{ref_p:>11.6f}"
              f"{diff:>12.2e}  {'OK ' if ok else 'FAIL':<5}  {snippet}")

    if failures:
        raise SystemExit(
            f"PARITY FAILED on {failures}/{PARITY_N} messages — JSON not written."
        )
    print(f"  All {PARITY_N} match within {PARITY_TOL:g}.")


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}")
    print(f"Loading {MODEL_PATH.relative_to(PROJECT_ROOT)} ...")
    model = joblib.load(MODEL_PATH)

    payload = build_payload(model)
    run_parity(model, payload)  # raises before any write if mismatch

    OUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    size = OUT_PATH.stat().st_size
    vocab_size = payload["feature_order"]["blocks"][0]["size"]
    n_struct = payload["feature_order"]["blocks"][1]["size"]
    print(f"\nWrote {OUT_PATH.relative_to(PROJECT_ROOT)} "
          f"({size:,} bytes, {size / 1_048_576:.2f} MiB)")
    print(
        f"Summary: vocab={vocab_size}, structural_features={n_struct}, "
        f"classes={payload['classifier']['classes']}, "
        f"threshold={payload['classifier']['threshold']}, "
        f"positive_class={payload['classifier']['positive_class']!r}"
    )


if __name__ == "__main__":
    main()
