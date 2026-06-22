"""Tests for src.explain — coefficient extraction, word mapping, explain()."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.explain import (
    explain,
    get_coefficients,
    structural_weights,
    top_terms_per_class,
)
from src.features import STRUCT_FEATURE_NAMES, build_feature_transformer


@pytest.fixture(scope="module")
def fitted_pipe() -> Pipeline:
    """Small LogReg pipeline with a deterministic spam marker term."""
    rows = []
    for i in range(10):
        rows.append(
            {
                "text": f"hello friend lunch today {i}",
                "label": "ham",
                "has_url": False,
                "has_phone": False,
                "has_email": False,
            }
        )
        # 'freebrand' appears ONLY in spam -> must get a positive spam coef.
        rows.append(
            {
                "text": f"freebrand offer discount sale {i}",
                "label": "spam",
                "has_url": False,
                "has_phone": True,
                "has_email": False,
            }
        )
        rows.append(
            {
                "text": f"verify kyc otp blocked http://x.com {i}",
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
    pipe.fit(df[["text", "has_url", "has_phone", "has_email"]], df["label"])
    return pipe


def test_coefficient_shape_matches_classes_and_features(fitted_pipe):
    classes, names, coef = get_coefficients(fitted_pipe)
    assert len(classes) == 3
    assert coef.shape == (len(classes), len(names))


def test_top_terms_returns_pos_neg_per_class(fitted_pipe):
    terms = top_terms_per_class(fitted_pipe, k=5)
    assert set(terms) == {"ham", "spam", "smishing"}
    for parts in terms.values():
        assert {"positive", "negative"} == set(parts)
        assert len(parts["positive"]) == 5


def test_structural_weights_shape(fitted_pipe):
    weights = structural_weights(fitted_pipe)
    assert list(weights.index) == STRUCT_FEATURE_NAMES
    assert weights.shape[1] == 3  # one column per class


def test_word_mapping_spam_marker_has_positive_weight(fitted_pipe):
    """A term seen only in spam must map to a positive spam coefficient."""
    terms = top_terms_per_class(fitted_pipe, k=50)
    spam_pos = terms["spam"]["positive"]
    marker = spam_pos[spam_pos["term"] == "freebrand"]
    assert not marker.empty
    assert marker.iloc[0]["coef"] > 0


def test_explain_returns_ranked_contributions(fitted_pipe):
    result = explain("verify kyc otp blocked http://x.com", pipe=fitted_pipe, k=5)
    assert result["predicted"] in {"ham", "spam", "smishing"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6
    assert len(result["top_contributions"]) > 0
    first = result["top_contributions"][0]
    assert {"feature", "kind", "value", "contribution"} == set(first)
