"""Phase 4 (Stage 2) — URL-feature ablation runner.

Compares the Stage-1 message classifier (TF-IDF + structural features) against
Stage-1 + lexical URL features, using the SAME LogisticRegression, seed, split,
and 5-fold CV protocol. Prints the ablation table (CV + held-out test), states
whether URL features helped, and shows an ``explain()`` example on a URL-bearing
message using the augmented model.

Stage 1 remains the primary, standalone result: this does NOT modify the
production model or any prior artifact. Network-based host features (WHOIS
domain age, certificate age, ASN/country) are future work — out of scope here.

Run: ``python -m src.ablation``.  Classical ML only; no neural nets.
"""

from __future__ import annotations

from src.data import load_processed
from src.explain import explain
from src.train import URL_AUGMENTED_MODEL_PATH, run_url_ablation


def _print_per_class(name: str, res: dict) -> None:
    rep = res["per_class_report"]
    print(f"\n  {name} — held-out test per-class:")
    print(f"    {'class':<10}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
    for cls in res["class_order"]:
        r = rep[cls]
        print(
            f"    {cls:<10}{r['precision']:>8.3f}{r['recall']:>8.3f}"
            f"{r['f1-score']:>8.3f}{int(r['support']):>9}"
        )


def main() -> None:
    print("=" * 72)
    print("Phase 4 — URL-feature ablation (Stage 1 vs Stage 1 + URL)")
    print("=" * 72)

    ablation, detail = run_url_ablation()

    print("\nAblation table (same LogReg, seed, split, 5-fold CV):")
    print(ablation.round(4).to_string())

    s1 = ablation.loc["stage1 (tfidf+struct)"]
    s2 = ablation.loc["stage1+url"]
    d_cv = s2["cv_macro_f1_mean"] - s1["cv_macro_f1_mean"]
    d_test = s2["test_macro_f1"] - s1["test_macro_f1"]
    d_smish = s2["test_f1_smishing"] - s1["test_f1_smishing"]

    print("\nDeltas (stage1+url minus stage1):")
    print(f"    CV macro-F1:        {d_cv:+.4f}")
    print(f"    test macro-F1:      {d_test:+.4f}")
    print(f"    test smishing F1:   {d_smish:+.4f}")

    # Honest verdict — URLs appear in only ~3.5% of messages.
    _, test_df = load_processed()
    url_rate = 100.0 * test_df["has_url"].mean()
    if d_cv > 0.002 and d_test >= 0:
        verdict = "URL features HELP (small but positive)."
    elif abs(d_cv) <= 0.002 and abs(d_test) <= 0.002:
        verdict = "URL features are NEUTRAL (no meaningful change)."
    else:
        verdict = "URL features do NOT help overall."
    print(f"\nVerdict: {verdict}")
    print(
        f"Context: URLs appear in only {url_rate:.1f}% of test messages, "
        "so any effect is expected to be small."
    )

    for name in ("stage1 (tfidf+struct)", "stage1+url"):
        _print_per_class(name, detail[name]["results"])

    # explain() on a URL-bearing message using the augmented model.
    url_msg = test_df[test_df["has_url"]].iloc[0]["text"]
    print("\n" + "=" * 72)
    print("explain() on a URL-bearing message (augmented stage1+url model)")
    print("=" * 72)
    exp = explain(url_msg, pipe=detail["stage1+url"]["pipe"], k=12)
    print(f"message:   {exp['message'][:120]}")
    print(f"predicted: {exp['predicted']}")
    print(
        "probabilities: "
        + ", ".join(f"{c}={p:.3f}" for c, p in exp["probabilities"].items())
    )
    print("top contributions (URL features marked 'url'):")
    print(f"    {'feature':<22}{'kind':<8}{'value':>8}{'contrib':>10}")
    for cc in exp["top_contributions"]:
        print(
            f"    {cc['feature'][:20]:<22}{cc['kind']:<8}"
            f"{cc['value']:>8.3f}{cc['contribution']:>10.3f}"
        )

    print(f"\nSaved augmented model: {URL_AUGMENTED_MODEL_PATH}")
    print("Production (Stage-1) model and all prior artifacts unchanged.")


if __name__ == "__main__":
    main()
