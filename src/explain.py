"""Phase 3 — interpretability for the Stage-1 LogisticRegression classifier.

Two levels of explanation, both from the linear model's signed coefficients
(honest, no surrogate):

* **Global** — rank coefficients per class. TF-IDF coefficients map back to the
  actual words/n-grams (top positive and negative terms per class); structural
  feature coefficients are reported with their signed weights. A
  permutation-importance pass on the structural features is saved as a
  cross-check. Tables -> ``paper/figures/*.csv``, plots -> ``paper/figures/*.png``.
* **Local** — :func:`explain` returns, for one message, the predicted class,
  class probabilities, and the top signed feature contributions (logit =
  coef * feature value). This feeds the demo's "why" panel later.

Run ``python -m src.explain`` to retrain+save the production LogReg model
(via :func:`src.train.train_production_model`), confirm its held-out metrics,
and regenerate every interpretability artifact. Classical ML only; no NN.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # file output only; no display / no frontend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from src.config import PROJECT_ROOT, get_config
from src.data import load_processed
from src.features import (
    STRUCT_FEATURE_NAMES,
    build_feature_transformer,
    compute_structural_features,
    message_to_frame,
)

FIGURES_DIR: Path = PROJECT_ROOT / "paper" / "figures"
TFIDF_PREFIX = "tfidf__"
STRUCT_PREFIX = "struct__"
URL_PREFIX = "url__"


# --- Model access ----------------------------------------------------------


def load_model(model_path: Path | None = None) -> Pipeline:
    """Load the fitted production pipeline (default: ``config.model_path``)."""
    path = get_config().model_path if model_path is None else model_path
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python -m src.train` / `src.explain`."
        )
    return joblib.load(path)


def get_coefficients(pipe: Pipeline) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (classes, feature_names, coef) for a fitted linear pipeline.

    ``coef`` has shape (n_classes, n_features), aligned with ``feature_names``
    and the classifier's own ``classes_`` order.
    """
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        raise TypeError("Interpretability layer requires a linear model (coef_).")
    feature_names = list(pipe.named_steps["features"].get_feature_names_out())
    return list(clf.classes_), feature_names, np.asarray(clf.coef_)


def _clean(name: str) -> tuple[str, str]:
    """Split a prefixed feature name into (kind, label)."""
    if name.startswith(TFIDF_PREFIX):
        return "term", name[len(TFIDF_PREFIX) :]
    if name.startswith(STRUCT_PREFIX):
        return "struct", name[len(STRUCT_PREFIX) :]
    if name.startswith(URL_PREFIX):
        return "url", name[len(URL_PREFIX) :]
    return "other", name


# --- Global interpretability ----------------------------------------------


def top_terms_per_class(
    pipe: Pipeline, k: int = 20
) -> dict[str, dict[str, pd.DataFrame]]:
    """Top-k positive and negative TF-IDF terms per class."""
    classes, names, coef = get_coefficients(pipe)
    term_idx = [i for i, n in enumerate(names) if n.startswith(TFIDF_PREFIX)]
    terms = [names[i][len(TFIDF_PREFIX) :] for i in term_idx]
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for c, cls in enumerate(classes):
        w = coef[c, term_idx]
        order = np.argsort(w)
        neg = pd.DataFrame(
            {"term": [terms[i] for i in order[:k]], "coef": w[order[:k]]}
        )
        pos = pd.DataFrame(
            {
                "term": [terms[i] for i in order[-k:][::-1]],
                "coef": w[order[-k:][::-1]],
            }
        )
        out[cls] = {"positive": pos, "negative": neg}
    return out


def structural_weights(pipe: Pipeline) -> pd.DataFrame:
    """Signed coefficient of each structural feature, per class (rows=feature)."""
    classes, names, coef = get_coefficients(pipe)
    struct_idx = [i for i, n in enumerate(names) if n.startswith(STRUCT_PREFIX)]
    labels = [names[i][len(STRUCT_PREFIX) :] for i in struct_idx]
    data = {cls: coef[c, struct_idx] for c, cls in enumerate(classes)}
    return pd.DataFrame(data, index=labels)


# --- Local interpretability ------------------------------------------------


def explain(message_text: str, pipe: Pipeline | None = None, k: int = 10) -> dict:
    """Explain one message: predicted class, probabilities, top contributions.

    Each contribution is the signed logit term (coef * feature value) for the
    predicted class — i.e. how much each word/structural signal pushed this
    message toward its predicted label.
    """
    if pipe is None:
        pipe = load_model()
    classes, names, coef = get_coefficients(pipe)

    frame = message_to_frame(message_text)
    x = pipe.named_steps["features"].transform(frame)
    x_row = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else x.ravel()

    proba = pipe.predict_proba(frame)[0]
    pred = str(pipe.predict(frame)[0])
    pred_idx = classes.index(pred)

    contrib = coef[pred_idx] * x_row
    active = np.nonzero(x_row)[0]
    ranked = sorted(active, key=lambda j: contrib[j], reverse=True)

    contributions = []
    for j in ranked[:k]:
        kind, label = _clean(names[j])
        contributions.append(
            {
                "feature": label,
                "kind": kind,
                "value": float(x_row[j]),
                "contribution": float(contrib[j]),
            }
        )
    return {
        "message": message_text,
        "predicted": pred,
        "probabilities": {c: float(p) for c, p in zip(classes, proba)},
        "top_contributions": contributions,
    }


# --- Permutation-importance cross-check (structural features) --------------


def permutation_importance_structural(
    seed: int | None = None, n_repeats: int = 10
) -> pd.DataFrame:
    """Permutation importance of each structural feature (cross-check).

    Fits a structural-features-only LogReg (so individual structural columns
    can be permuted) and measures the macro-F1 drop on the test set.
    """
    cfg = get_config()
    seed = cfg.random_seed if seed is None else seed
    cols = ["text", "has_url", "has_phone", "has_email"]
    train_df, test_df = load_processed()

    X_tr = compute_structural_features(train_df[cols])
    X_te = compute_structural_features(test_df[cols])
    model = Pipeline(
        [
            ("scale", MaxAbsScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=seed
                ),
            ),
        ]
    )
    model.fit(X_tr, train_df["label"])
    result = permutation_importance(
        model,
        X_te,
        test_df["label"],
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=seed,
    )
    return (
        pd.DataFrame(
            {
                "feature": STRUCT_FEATURE_NAMES,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


# --- Plotting --------------------------------------------------------------


def _plot_top_terms(cls: str, pos: pd.DataFrame, neg: pd.DataFrame, path: Path):
    df = pd.concat([neg, pos]).drop_duplicates("term")
    df = df.sort_values("coef")
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in df["coef"]]
    plt.figure(figsize=(8, max(4, 0.3 * len(df))))
    plt.barh(df["term"], df["coef"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"Top TF-IDF terms for class '{cls}' (LogReg coefficients)")
    plt.xlabel("coefficient (red = pushes away, green = pushes toward)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_structural_weights(weights: pd.DataFrame, path: Path):
    ax = weights.plot.barh(figsize=(9, 7))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Structural-feature coefficients per class (LogReg)")
    ax.set_xlabel("signed coefficient")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_permutation_importance(perm: pd.DataFrame, path: Path):
    plt.figure(figsize=(8, 6))
    plt.barh(
        perm["feature"],
        perm["importance_mean"],
        xerr=perm["importance_std"],
        color="#2980b9",
    )
    plt.gca().invert_yaxis()
    plt.title("Permutation importance of structural features (macro-F1 drop)")
    plt.xlabel("mean importance (decrease in macro-F1)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_global_artifacts(pipe: Pipeline, k: int = 20) -> None:
    """Write all global tables (CSV) and plots (PNG) to ``paper/figures/``."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    terms = top_terms_per_class(pipe, k=k)
    for cls, parts in terms.items():
        pos = parts["positive"].assign(direction="positive")
        neg = parts["negative"].assign(direction="negative")
        pd.concat([pos, neg]).to_csv(
            FIGURES_DIR / f"coef_top_terms_{cls}.csv", index=False
        )
        _plot_top_terms(
            cls,
            parts["positive"].head(15),
            parts["negative"].head(15),
            FIGURES_DIR / f"coef_top_terms_{cls}.png",
        )

    weights = structural_weights(pipe)
    weights.to_csv(FIGURES_DIR / "structural_weights.csv")
    _plot_structural_weights(weights, FIGURES_DIR / "structural_weights.png")

    perm = permutation_importance_structural()
    perm.to_csv(FIGURES_DIR / "permutation_importance_structural.csv", index=False)
    _plot_permutation_importance(
        perm, FIGURES_DIR / "permutation_importance_structural.png"
    )


# --- CLI -------------------------------------------------------------------


def main() -> None:
    from src.train import train_production_model

    print("=" * 70)
    print("Phase 3 — interpretability (production model: LogisticRegression)")
    print("=" * 70)

    pipe, results = train_production_model()
    print("\nProduction LogReg held-out TEST metrics (saved as production model):")
    print(f"  macro-F1:                 {results['macro_f1']:.4f}")
    print(
        "  binary (smishing vs not): "
        f"{results['binary_smishing_vs_not_macro_f1']:.4f}"
    )
    rep = results["per_class_report"]
    print(f"    {'class':<10}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
    for cls in results["class_order"]:
        r = rep[cls]
        print(
            f"    {cls:<10}{r['precision']:>8.3f}{r['recall']:>8.3f}"
            f"{r['f1-score']:>8.3f}{int(r['support']):>9}"
        )
    print("\n  Confusion matrix (rows=true, cols=pred):")
    order = results["class_order"]
    print("    true\\pred " + "".join(f"{c:>10}" for c in order))
    for cls, row in zip(order, results["confusion_matrix"]):
        print(f"    {cls:<9}" + "".join(f"{v:>10}" for v in row))

    save_global_artifacts(pipe)

    # Headline global result: top positive terms per class.
    print("\n" + "=" * 70)
    print("GLOBAL: top positive TF-IDF terms per class")
    print("=" * 70)
    terms = top_terms_per_class(pipe, k=12)
    classes = list(terms)
    width = 18
    print("".join(f"{c:<{width}}" for c in classes))
    for rank in range(12):
        line = ""
        for cls in classes:
            t = terms[cls]["positive"].iloc[rank]
            line += f"{t['term'][:14]:<12}{t['coef']:>5.1f} "
        print(line)

    print("\n" + "=" * 70)
    print("GLOBAL: structural-feature coefficients per class")
    print("=" * 70)
    print(structural_weights(pipe).round(3).to_string())

    print("\n" + "=" * 70)
    print("Permutation importance of structural features (cross-check)")
    print("=" * 70)
    print(permutation_importance_structural().round(4).to_string(index=False))

    # Sample local explanation on a real smishing message from the test set.
    _, test_df = load_processed()
    sample = test_df[test_df["label"] == "smishing"].iloc[0]["text"]
    print("\n" + "=" * 70)
    print("LOCAL: explain() on a sample smishing message")
    print("=" * 70)
    exp = explain(sample, pipe=pipe, k=10)
    print(f"message:   {exp['message'][:120]}")
    print(f"predicted: {exp['predicted']}")
    print(
        "probabilities: "
        + ", ".join(f"{c}={p:.3f}" for c, p in exp["probabilities"].items())
    )
    print("top contributions toward predicted class:")
    print(f"    {'feature':<22}{'kind':<8}{'value':>8}{'contrib':>10}")
    for cc in exp["top_contributions"]:
        print(
            f"    {cc['feature'][:20]:<22}{cc['kind']:<8}"
            f"{cc['value']:>8.3f}{cc['contribution']:>10.3f}"
        )

    print(f"\nArtifacts written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
