"""Tests for src.train — pipelines, CV, evaluation, and model round-trip."""

import joblib
import pandas as pd
import pytest

from src.features import build_feature_transformer
from src.train import (
    CLASS_ORDER,
    build_models,
    evaluate_on_test,
    logreg_pipeline,
    run_cv,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def _labeled_df(n_per_class: int = 8) -> pd.DataFrame:
    """Small but CV-able (>=5 per class) synthetic 3-class dataset."""
    rows = []
    for i in range(n_per_class):
        rows.append(
            {
                "text": f"hi friend how are you today {i}",
                "label": "ham",
                "has_url": False,
                "has_phone": False,
                "has_email": False,
            }
        )
        rows.append(
            {
                "text": f"free offer discount sale buy now {i}",
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
    return pd.DataFrame(rows)


def _nb_pipe() -> Pipeline:
    return Pipeline(
        [("features", build_feature_transformer(min_df=1)), ("clf", MultinomialNB())]
    )


def test_build_models_returns_four_pipelines():
    models = build_models(seed=42)
    assert set(models) == {
        "MultinomialNB",
        "LogisticRegression",
        "RandomForest",
        "XGBoost",
    }
    assert all(isinstance(p, Pipeline) for p in models.values())


def test_pipeline_fits_and_predicts():
    df = _labeled_df()
    pipe = _nb_pipe()
    pipe.fit(df[["text", "has_url", "has_phone", "has_email"]], df["label"])
    preds = pipe.predict(df[["text", "has_url", "has_phone", "has_email"]])
    assert len(preds) == len(df)
    assert set(preds).issubset(set(CLASS_ORDER))


def test_run_cv_on_train_returns_expected_columns():
    df = _labeled_df()
    X = df[["text", "has_url", "has_phone", "has_email"]]
    y = df["label"]
    models = {"MultinomialNB": _nb_pipe()}
    table = run_cv(models, X, y, seed=42)
    assert "macro_f1_mean" in table.columns
    assert "macro_f1_std" in table.columns
    for cls in CLASS_ORDER:
        assert f"f1_{cls}" in table.columns
    assert "MultinomialNB" in table.index


def test_evaluate_on_test_returns_metrics():
    df = _labeled_df()
    cols = ["text", "has_url", "has_phone", "has_email"]
    results = evaluate_on_test(_nb_pipe(), df[cols], df["label"], df[cols], df["label"])
    assert 0.0 <= results["macro_f1"] <= 1.0
    assert len(results["confusion_matrix"]) == len(CLASS_ORDER)
    assert "binary_smishing_vs_not_macro_f1" in results


def test_saved_model_round_trips(tmp_path):
    df = _labeled_df()
    cols = ["text", "has_url", "has_phone", "has_email"]
    pipe = _nb_pipe()
    pipe.fit(df[cols], df["label"])
    before = pipe.predict(df[cols])

    path = tmp_path / "model.pkl"
    joblib.dump(pipe, path)
    loaded = joblib.load(path)
    after = loaded.predict(df[cols])

    assert list(before) == list(after)


def test_augmented_pipeline_fits_and_predicts():
    df = _labeled_df()
    cols = ["text", "has_url", "has_phone", "has_email"]
    pipe = logreg_pipeline(seed=42, include_url=True)
    pipe.fit(df[cols], df["label"])
    preds = pipe.predict(df[cols])
    assert len(preds) == len(df)
    assert set(preds).issubset(set(CLASS_ORDER))


def test_augmented_model_round_trips(tmp_path):
    df = _labeled_df()
    cols = ["text", "has_url", "has_phone", "has_email"]
    pipe = logreg_pipeline(seed=42, include_url=True)
    pipe.fit(df[cols], df["label"])
    before = pipe.predict(df[cols])

    path = tmp_path / "aug_model.pkl"
    joblib.dump(pipe, path)
    after = joblib.load(path).predict(df[cols])
    assert list(before) == list(after)
