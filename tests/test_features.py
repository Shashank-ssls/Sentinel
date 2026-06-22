"""Tests for src.features — structural + URL features and combined transformer."""

import numpy as np
import pandas as pd

from src.features import (
    FEATURE_COLUMNS,
    STRUCT_FEATURE_NAMES,
    URL_FEATURE_NAMES,
    build_feature_transformer,
    compute_structural_features,
    compute_url_features,
    derive_flags,
    message_to_frame,
)


def _tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["FREE OTP win £5", "hello how are you", "call me later"],
            "has_url": [True, False, False],
            "has_phone": [False, False, True],
            "has_email": [False, False, False],
        }
    )


def test_structural_feature_values_are_correct():
    df = _tiny_df().iloc[[0]]
    feats = compute_structural_features(df)
    assert feats.shape == (1, len(STRUCT_FEATURE_NAMES))
    row = dict(zip(STRUCT_FEATURE_NAMES, feats[0]))
    assert row["message_length"] == 15.0
    assert row["word_count"] == 4.0
    assert row["digit_count"] == 1.0
    assert row["uppercase_ratio"] == 0.7  # 7 of 10 letters uppercase
    assert row["count_links"] == 0.0
    assert row["has_url"] == 1.0
    assert row["has_phone"] == 0.0
    assert row["count_currency"] == 1.0
    assert row["urgency_count"] == 3.0  # free, otp, win


def test_structural_features_are_non_negative():
    feats = compute_structural_features(_tiny_df())
    assert (feats >= 0).all()


def test_combined_transformer_fit_transform_shapes():
    df = _tiny_df()
    transformer = build_feature_transformer(min_df=1)
    matrix = transformer.fit_transform(df)
    assert matrix.shape[0] == len(df)
    # TF-IDF vocab + the 10 structural columns -> more than 10 columns.
    assert matrix.shape[1] > len(STRUCT_FEATURE_NAMES)


def test_derive_flags_detects_url_email_phone():
    assert derive_flags("visit http://x.com now")["has_url"] is True
    assert derive_flags("mail me a@b.com")["has_email"] is True
    assert derive_flags("ring +1 234 567 8901")["has_phone"] is True
    assert derive_flags("plain text")["has_url"] is False


def test_message_to_frame_has_feature_columns():
    frame = message_to_frame("hello http://x.com")
    assert list(frame.columns) == FEATURE_COLUMNS
    assert len(frame) == 1
    assert bool(frame.loc[0, "has_url"]) is True


# --- URL features (Stage 2) ------------------------------------------------


def _url_df(text: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"text": [text], "has_url": [True], "has_phone": [False], "has_email": [False]}
    )


def test_url_features_extract_correctly():
    df = _url_df("Verify: http://sbi-secure-login.xyz/account@x")
    feats = compute_url_features(df)
    assert feats.shape == (1, len(URL_FEATURE_NAMES))
    row = dict(zip(URL_FEATURE_NAMES, feats[0]))
    assert row["url_length"] > 0
    assert row["url_count_dots"] >= 1
    assert row["url_has_at"] == 1.0
    assert row["url_suspicious_tld"] == 1.0  # .xyz
    assert row["url_is_https"] == 0.0  # http
    assert row["url_is_ip_host"] == 0.0
    # 'sbi-secure-login' resembles the protected brand 'sbi' -> some similarity.
    assert 0.0 < row["url_brand_similarity"] <= 1.0


def test_url_features_detect_ip_and_shortener_and_https():
    ip = dict(
        zip(URL_FEATURE_NAMES, compute_url_features(_url_df("http://192.168.0.1/a"))[0])
    )
    assert ip["url_is_ip_host"] == 1.0
    short = dict(
        zip(URL_FEATURE_NAMES, compute_url_features(_url_df("see www.bit.ly/xyz"))[0])
    )
    assert short["url_is_shortener"] == 1.0
    https = dict(
        zip(
            URL_FEATURE_NAMES,
            compute_url_features(_url_df("https://www.hdfcbank.com"))[0],
        )
    )
    assert https["url_is_https"] == 1.0


def test_url_features_neutral_when_no_url():
    df = pd.DataFrame(
        {
            "text": ["just a plain message"],
            "has_url": [False],
            "has_phone": [False],
            "has_email": [False],
        }
    )
    feats = compute_url_features(df)
    assert feats.shape == (1, len(URL_FEATURE_NAMES))
    assert (feats == 0.0).all()


def test_augmented_transformer_adds_url_block():
    df = _url_df("win prize http://sbi-secure.xyz")
    base = build_feature_transformer(min_df=1).fit_transform(df)
    augmented = build_feature_transformer(min_df=1, include_url=True).fit_transform(df)
    assert augmented.shape[1] == base.shape[1] + len(URL_FEATURE_NAMES)
