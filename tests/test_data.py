"""Tests for src.data — the Phase 1 smishing data pipeline."""

import pandas as pd
import pytest

from src.data import (
    LABELS,
    SCHEMA,
    _normalize_label,
    _yesno_to_bool,
    load_raw,
    make_split,
)


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """The normalized full dataset (loaded once per module)."""
    return load_raw()


# --- Schema correctness ----------------------------------------------------


def test_schema_columns_exact_and_ordered(raw_df):
    assert list(raw_df.columns) == SCHEMA


def test_flag_columns_are_boolean(raw_df):
    for flag in ("has_url", "has_phone", "has_email", "india_relevant"):
        assert raw_df[flag].dtype == bool


def test_source_and_india_defaults(raw_df):
    assert (raw_df["source"] == "mendeley").all()
    assert (~raw_df["india_relevant"]).all()


def test_message_ids_unique(raw_df):
    assert raw_df["message_id"].is_unique


# --- Label normalization ---------------------------------------------------


def test_exactly_three_lowercase_classes(raw_df):
    present = set(raw_df["label"].unique())
    assert present == set(LABELS)
    assert all(lbl == lbl.lower() for lbl in present)


def test_normalize_label_handles_mixed_case():
    assert _normalize_label("Smishing") == "smishing"
    assert _normalize_label("Spam") == "spam"
    assert _normalize_label("ham") == "ham"


def test_normalize_label_rejects_unknown():
    with pytest.raises(ValueError):
        _normalize_label("phishing")


def test_yesno_to_bool_case_insensitive():
    assert _yesno_to_bool("yes") is True
    assert _yesno_to_bool("No") is False
    with pytest.raises(ValueError):
        _yesno_to_bool("maybe")


# --- Split reproducibility & leakage --------------------------------------


def test_split_is_reproducible(raw_df):
    train_a, test_a = make_split(raw_df, seed=42)
    train_b, test_b = make_split(raw_df, seed=42)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)


def test_split_covers_all_rows_without_loss(raw_df):
    train_df, test_df = make_split(raw_df, seed=42)
    assert len(train_df) + len(test_df) == len(raw_df)


def test_no_message_id_leakage_across_splits(raw_df):
    train_df, test_df = make_split(raw_df, seed=42)
    overlap = set(train_df["message_id"]) & set(test_df["message_id"])
    assert overlap == set()


def test_no_text_leakage_across_splits(raw_df):
    """Stronger guard: no identical text may appear in both train and test."""
    train_df, test_df = make_split(raw_df, seed=42)
    text_overlap = set(train_df["text"]) & set(test_df["text"])
    assert text_overlap == set()


def test_load_raw_is_text_deduplicated_by_default(raw_df):
    """Default load is deduped on text; dedup=False preserves raw rows."""
    assert not raw_df["text"].duplicated().any()
    raw_all = load_raw(dedup=False)
    assert len(raw_all) > len(raw_df)  # duplicates exist and were dropped


def test_split_is_stratified(raw_df):
    train_df, test_df = make_split(raw_df, seed=42)
    for label in LABELS:
        full = (raw_df["label"] == label).mean()
        tr = (train_df["label"] == label).mean()
        te = (test_df["label"] == label).mean()
        assert abs(tr - full) < 0.01
        assert abs(te - full) < 0.01
