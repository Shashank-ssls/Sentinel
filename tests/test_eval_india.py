"""Tests for src.eval_india — cleaning, binary mapping, evaluation-only runs."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.eval_india import (
    build_eval_frame,
    clean_real,
    evaluate_real,
    evaluate_synthetic,
    to_unwanted_legit,
)
from src.features import FEATURE_COLUMNS, build_feature_transformer
from src.train import CLASS_ORDER


@pytest.fixture(scope="module")
def fitted_pipe() -> Pipeline:
    rows = []
    for i in range(8):
        rows.append(
            {
                "text": f"hi friend lunch today {i}",
                "label": "ham",
                "has_url": False,
                "has_phone": False,
                "has_email": False,
            }
        )
        rows.append(
            {
                "text": f"free offer discount sale {i}",
                "label": "spam",
                "has_url": False,
                "has_phone": True,
                "has_email": False,
            }
        )
        rows.append(
            {
                "text": f"verify kyc otp http://x.com {i}",
                "label": "smishing",
                "has_url": True,
                "has_phone": False,
                "has_email": False,
            }
        )
    df = pd.DataFrame(rows)
    pipe = Pipeline(
        [
            ("features", build_feature_transformer(min_df=1)),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    pipe.fit(df[FEATURE_COLUMNS], df["label"])
    return pipe


def test_clean_real_removes_right_rows():
    df = pd.DataFrame(
        {
            "Msg": [
                "Win a prize now",  # keep
                "Win a prize now",  # exact dup -> removed
                "image omitted",  # junk -> removed
                None,  # null -> removed
                "  ",  # empty -> removed
                "Legit hello there",  # keep
            ],
            "Label": ["spam", "spam", "ham", "ham", "ham", "ham"],
        }
    )
    clean, summary = clean_real(df)
    assert summary["removed_null_or_empty"] == 2
    assert summary["removed_image_omitted"] == 1
    assert summary["removed_duplicate_text"] == 1
    assert summary["final_rows"] == 2
    assert list(clean.columns) == ["text", "label"]


def test_to_unwanted_legit_mapping():
    out = to_unwanted_legit(["ham", "spam", "smishing", "HAM"])
    assert list(out) == ["legit", "unwanted", "unwanted", "legit"]


def test_build_eval_frame_derives_flags():
    frame = build_eval_frame(["plain text", "go http://x.com now"])
    assert list(frame.columns) == FEATURE_COLUMNS
    assert bool(frame.loc[1, "has_url"]) is True
    assert bool(frame.loc[0, "has_url"]) is False


def test_evaluate_real_runs_on_injected_pipe(fitted_pipe):
    clean = pd.DataFrame(
        {
            "text": ["free prize claim now", "hi see you at lunch"],
            "label": ["spam", "ham"],
        }
    )
    res = evaluate_real(fitted_pipe, clean)
    assert 0.0 <= res["binary_macro_f1"] <= 1.0
    assert res["labels"] == ["legit", "unwanted"]
    assert len(res["confusion_matrix"]) == 2


def test_evaluate_synthetic_runs_on_injected_pipe(fitted_pipe):
    df = pd.DataFrame(
        {
            "text": [
                "account statement ready",
                "free offer now",
                "verify kyc otp http://x.com",
            ],
            "label": ["ham", "spam", "smishing"],
            "notes": ["bank", "promo", "upi phishing"],
        }
    )
    res = evaluate_synthetic(fitted_pipe, df)
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert res["labels"] == list(CLASS_ORDER)
    assert len(res["confusion_matrix"]) == 3
