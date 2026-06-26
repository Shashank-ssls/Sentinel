"""Tests for src.india_augment — split integrity, binary mapping, reproducibility."""

import numpy as np
import pandas as pd

from src.eval_india import to_unwanted_legit
from src.features import FEATURE_COLUMNS
from src.india_augment import (
    evaluate_binary,
    make_india_split,
    train_binary_model,
)


def _clean_india(n_per_class: int = 12) -> pd.DataFrame:
    rows = []
    for i in range(n_per_class):
        rows.append(
            {"text": f"dear customer your account statement {i}", "label": "ham"}
        )
        rows.append({"text": f"win free recharge offer claim now {i}", "label": "spam"})
    return pd.DataFrame(rows)


def test_india_split_no_leakage_and_reproducible():
    df = _clean_india()
    tr1, te1 = make_india_split(df, test_size=0.3, seed=42)
    tr2, te2 = make_india_split(df, test_size=0.3, seed=42)

    # No text overlap between train and test (sacred test set).
    assert set(tr1["text"]) & set(te1["text"]) == set()
    # Covers all rows, reproducible.
    assert len(tr1) + len(te1) == len(df)
    pd.testing.assert_frame_equal(tr1, tr2)
    pd.testing.assert_frame_equal(te1, te2)


def test_india_split_is_stratified():
    df = _clean_india()
    tr, te = make_india_split(df, test_size=0.3, seed=42)
    for lbl in ("ham", "spam"):
        full = (df["label"] == lbl).mean()
        assert abs((te["label"] == lbl).mean() - full) < 0.1


def test_binary_mapping_correct():
    out = to_unwanted_legit(["ham", "spam", "smishing"])
    assert list(out) == ["legit", "unwanted", "unwanted"]


def _train_frame(n: int = 10) -> tuple[pd.DataFrame, np.ndarray]:
    rows, labels = [], []
    for i in range(n):
        rows.append(
            {
                "text": f"hi friend lunch plan {i}",
                "has_url": False,
                "has_phone": False,
                "has_email": False,
            }
        )
        labels.append("legit")
        rows.append(
            {
                "text": f"win free prize claim now upi {i}",
                "has_url": False,
                "has_phone": True,
                "has_email": False,
            }
        )
        labels.append("unwanted")
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS), np.array(labels)


def test_models_train_and_evaluate_reproducibly():
    X, y = _train_frame()
    m1 = train_binary_model(X, y, seed=42)
    m2 = train_binary_model(X, y, seed=42)
    p1 = m1.predict(X)
    p2 = m2.predict(X)
    assert list(p1) == list(p2)  # deterministic

    res = evaluate_binary(m1, X, y)
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert len(res["confusion_matrix"]) == 2
