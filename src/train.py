"""Phase 2 — model comparison + final evaluation (Stage 1 message classifier).

Compares four classical models (Multinomial Naive Bayes, Logistic Regression,
Random Forest, XGBoost), each as a Pipeline whose first step is the shared
feature transformer from :mod:`src.features` (so TF-IDF is fit inside every CV
fold — no leakage). Selection is by mean macro-F1 over stratified 5-fold CV on
the TRAINING set only; the held-out test set is touched exactly once, at the end.

Artifacts written under ``models/``:
  * ``cv_comparison.csv``   — per-model CV macro-F1 (+/- std) and per-class F1
  * ``test_results.json``   — chosen model's held-out metrics
  * ``confusion_matrix.csv``— chosen model's test confusion matrix
  * best fitted pipeline    — serialized to ``config.model_path``

Run: ``python -m src.train``.  No URL logic, no neural nets (by design).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.config import PROJECT_ROOT, get_config
from src.data import load_processed
from src.features import build_feature_transformer

# Fixed class order for confusion matrices / per-class reports.
CLASS_ORDER: tuple[str, ...] = ("ham", "spam", "smishing")

MODELS_DIR: Path = PROJECT_ROOT / "models"
CV_PATH: Path = MODELS_DIR / "cv_comparison.csv"
TEST_RESULTS_PATH: Path = MODELS_DIR / "test_results.json"
CONFUSION_PATH: Path = MODELS_DIR / "confusion_matrix.csv"

# Production model (Phase 3 decision): LogisticRegression is chosen over the
# marginally-higher RandomForest because its signed coefficients give direct,
# honest explanations and the two are within CV variance. RF artifacts are
# preserved (RF pickle copied aside; its comparison/test artifacts untouched).
PRODUCTION_MODEL: str = "LogisticRegression"
RF_MODEL_PATH: Path = MODELS_DIR / "randomforest_model.pkl"
PRODUCTION_TEST_RESULTS_PATH: Path = MODELS_DIR / "production_test_results.json"
PRODUCTION_CONFUSION_PATH: Path = MODELS_DIR / "production_confusion_matrix.csv"

# Phase 4 (Stage 2) — URL-feature ablation artifacts.
URL_AUGMENTED_MODEL_PATH: Path = MODELS_DIR / "url_augmented_model.pkl"
FIGURES_DIR: Path = PROJECT_ROOT / "paper" / "figures"
ABLATION_PATH: Path = FIGURES_DIR / "url_ablation.csv"

FEATURE_COLS: list[str] = ["text", "has_url", "has_phone", "has_email"]

N_SPLITS: int = 5


def logreg_pipeline(seed: int, include_url: bool = False) -> Pipeline:
    """LogReg production pipeline, optionally with the Stage-2 URL block."""
    return Pipeline(
        [
            ("features", build_feature_transformer(include_url=include_url)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=seed
                ),
            ),
        ]
    )


class XGBStringClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost wrapper that accepts string labels (encodes/decodes internally).

    XGBoost requires contiguous integer labels; this keeps the rest of the
    pipeline working with the human-readable class names.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        random_state: int = 42,
        n_jobs: int = -1,
        tree_method: str = "hist",
        eval_metric: str = "mlogloss",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tree_method = tree_method
        self.eval_metric = eval_metric

    def fit(self, X, y):
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method=self.tree_method,
            eval_metric=self.eval_metric,
            objective="multi:softprob",
        )
        self.model_.fit(X, self.le_.transform(y))
        return self

    def predict(self, X):
        return self.le_.inverse_transform(self.model_.predict(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def build_models(seed: int) -> dict[str, Pipeline]:
    """Return the four model pipelines, each with its own feature step.

    Class imbalance is handled via ``class_weight="balanced"`` where supported.
    MultinomialNB and XGBoost (multiclass) do not accept ``class_weight``; the
    macro-F1 headline metric keeps the comparison fair despite the ~81% ham
    majority.
    """
    specs = {
        "MultinomialNB": MultinomialNB(),  # no class_weight support
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "XGBoost": XGBStringClassifier(random_state=seed),  # no class_weight
    }
    return {
        name: Pipeline([("features", build_feature_transformer()), ("clf", clf)])
        for name, clf in specs.items()
    }


def _scorers() -> dict:
    """macro-F1 plus a per-class F1 scorer for each class."""
    scoring = {"f1_macro": "f1_macro"}
    for cls in CLASS_ORDER:
        scoring[f"f1_{cls}"] = make_scorer(
            f1_score, labels=[cls], average="macro", zero_division=0
        )
    return scoring


def run_cv(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> pd.DataFrame:
    """Stratified 5-fold CV on the TRAINING data only; one row per model."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    scoring = _scorers()
    rows = []
    for name, pipe in models.items():
        res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
        row = {
            "model": name,
            "macro_f1_mean": res["test_f1_macro"].mean(),
            "macro_f1_std": res["test_f1_macro"].std(),
        }
        for cls in CLASS_ORDER:
            row[f"f1_{cls}"] = res[f"test_f1_{cls}"].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def evaluate_on_test(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Retrain on full train, predict test ONCE, return all metrics."""
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_test,
        y_pred,
        labels=list(CLASS_ORDER),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(CLASS_ORDER))

    # Collapsed binary (smishing vs not) macro-F1 — free extra.
    to_bin = lambda arr: np.where(np.asarray(arr) == "smishing", "smishing", "not")
    binary_macro_f1 = f1_score(
        to_bin(y_test), to_bin(y_pred), average="macro", zero_division=0
    )

    return {
        "macro_f1": macro_f1,
        "binary_smishing_vs_not_macro_f1": binary_macro_f1,
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
        "class_order": list(CLASS_ORDER),
    }


def _print_confusion(cm: list[list[int]]) -> None:
    header = "true\\pred  " + "".join(f"{c:>10}" for c in CLASS_ORDER)
    print(header)
    for cls, row in zip(CLASS_ORDER, cm):
        print(f"{cls:<10}" + "".join(f"{v:>10}" for v in row))


def train_production_model(
    model_name: str = PRODUCTION_MODEL, seed: int | None = None
) -> tuple[Pipeline, dict]:
    """Retrain the chosen model on full train, evaluate once, save as production.

    Saves the fitted pipeline to ``config.model_path`` and writes
    ``production_test_results.json`` + ``production_confusion_matrix.csv``.
    The existing RandomForest artifacts are preserved: the prior production
    pickle (RF, from Phase 2) is copied to ``randomforest_model.pkl`` before
    being overwritten, and the RF comparison/test artifacts are left untouched.
    """
    cfg = get_config()
    seed = cfg.random_seed if seed is None else seed
    np.random.seed(seed)

    train_df, test_df = load_processed()
    X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["label"]

    pipe = build_models(seed)[model_name]
    results = evaluate_on_test(pipe, X_train, y_train, X_test, y_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Preserve the Phase 2 RandomForest pickle before overwriting the path.
    if cfg.model_path.exists() and not RF_MODEL_PATH.exists():
        shutil.copy(cfg.model_path, RF_MODEL_PATH)

    joblib.dump(pipe, cfg.model_path)
    pd.DataFrame(
        results["confusion_matrix"],
        index=[f"true_{c}" for c in CLASS_ORDER],
        columns=[f"pred_{c}" for c in CLASS_ORDER],
    ).to_csv(PRODUCTION_CONFUSION_PATH)
    with open(PRODUCTION_TEST_RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {"production_model": model_name, "seed": seed, **results}, fh, indent=2
        )
    return pipe, results


def run_url_ablation(seed: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Compare Stage-1 (TF-IDF+structural) vs Stage-1+URL, same LogReg/seed/CV.

    Returns (ablation_table, detail) where detail holds the per-config test
    results and the fitted augmented pipeline. Saves the table to
    ``paper/figures/url_ablation.csv`` and the augmented model to
    ``models/url_augmented_model.pkl``. Does NOT touch the Stage-1 production
    model or any prior artifact.
    """
    cfg = get_config()
    seed = cfg.random_seed if seed is None else seed
    np.random.seed(seed)

    train_df, test_df = load_processed()
    X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["label"]

    configs = {
        "stage1 (tfidf+struct)": False,
        "stage1+url": True,
    }
    cv_table = run_cv(
        {name: logreg_pipeline(seed, inc) for name, inc in configs.items()},
        X_train,
        y_train,
        seed,
    )

    rows, detail = [], {}
    for name, include_url in configs.items():
        pipe = logreg_pipeline(seed, include_url=include_url)
        res = evaluate_on_test(pipe, X_train, y_train, X_test, y_test)
        detail[name] = {"results": res, "pipe": pipe}
        rep = res["per_class_report"]
        rows.append(
            {
                "config": name,
                "cv_macro_f1_mean": cv_table.loc[name, "macro_f1_mean"],
                "cv_macro_f1_std": cv_table.loc[name, "macro_f1_std"],
                "test_macro_f1": res["macro_f1"],
                "test_f1_ham": rep["ham"]["f1-score"],
                "test_f1_spam": rep["spam"]["f1-score"],
                "test_f1_smishing": rep["smishing"]["f1-score"],
                "test_binary_macro_f1": res["binary_smishing_vs_not_macro_f1"],
            }
        )
    ablation = pd.DataFrame(rows).set_index("config")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ablation.to_csv(ABLATION_PATH)
    joblib.dump(detail["stage1+url"]["pipe"], URL_AUGMENTED_MODEL_PATH)
    return ablation, detail


def main() -> None:
    cfg = get_config()
    seed = cfg.random_seed
    np.random.seed(seed)

    train_df, test_df = load_processed()
    feat_cols = ["text", "has_url", "has_phone", "has_email"]
    X_train, y_train = train_df[feat_cols], train_df["label"]
    X_test, y_test = test_df[feat_cols], test_df["label"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Phase 2 — model comparison ({N_SPLITS}-fold CV on train, seed={seed})")
    print(f"train={len(train_df)}  test={len(test_df)}  (test untouched in CV)")
    print("=" * 70)

    models = build_models(seed)
    cv_table = run_cv(models, X_train, y_train, seed)
    cv_table_sorted = cv_table.sort_values("macro_f1_mean", ascending=False)
    cv_table_sorted.to_csv(CV_PATH)

    print("\nCV comparison (mean macro-F1 +/- std; per-class F1 means):")
    print(cv_table_sorted.round(4).to_string())

    best_name = cv_table_sorted.index[0]
    print(f"\nBest model by mean macro-F1: {best_name}")

    best_pipe = build_models(seed)[best_name]
    results = evaluate_on_test(best_pipe, X_train, y_train, X_test, y_test)

    print("\n" + "=" * 70)
    print(f"Held-out TEST results — {best_name} (evaluated once)")
    print("=" * 70)
    print(f"macro-F1:                 {results['macro_f1']:.4f}")
    print(
        "binary (smishing vs not): " f"{results['binary_smishing_vs_not_macro_f1']:.4f}"
    )
    print("\nPer-class precision / recall / F1:")
    rep = results["per_class_report"]
    print(f"    {'class':<10}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
    for cls in CLASS_ORDER:
        r = rep[cls]
        print(
            f"    {cls:<10}{r['precision']:>8.3f}{r['recall']:>8.3f}"
            f"{r['f1-score']:>8.3f}{int(r['support']):>9}"
        )
    print("\nConfusion matrix (rows=true, cols=pred):")
    _print_confusion(results["confusion_matrix"])

    # Persist artifacts.
    pd.DataFrame(
        results["confusion_matrix"],
        index=[f"true_{c}" for c in CLASS_ORDER],
        columns=[f"pred_{c}" for c in CLASS_ORDER],
    ).to_csv(CONFUSION_PATH)
    with open(TEST_RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"chosen_model": best_name, "seed": seed, **results}, fh, indent=2)
    joblib.dump(best_pipe, cfg.model_path)

    print("\nArtifacts written:")
    print(f"  {CV_PATH}")
    print(f"  {TEST_RESULTS_PATH}")
    print(f"  {CONFUSION_PATH}")
    print(f"  {cfg.model_path}  (fitted pipeline: vectorizer + model)")


if __name__ == "__main__":
    main()
