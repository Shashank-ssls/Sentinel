# Sentinel — Paper Outline (IEEE conference style)

> Working skeleton. Sections fill with real numbers/tables/figures as phases
> complete. Target venue: TBD (student conference / IEEE workshop / IJRASET).

**Working title:** Interpretable Classical-ML Phishing-Link Detection with an
India-Focused Evaluation

**Research question:** Can interpretable lexical + host-based features achieve
phishing-detection performance competitive with heavier models while remaining
fully explainable — and how well does such a model generalize to India-relevant
scams (UPI, bank-impersonation, fake-KYC)?

---

## Abstract
Problem, approach (interpretable classical ML), key result, India angle. *(fill last)*

## 1. Introduction
- Phishing scale and the India context (UPI/bank-impersonation/KYC scams).
- The interpretability gap in ML-based detection.
- Contributions (bulleted): interpretable feature set; model comparison;
  feature-importance analysis; India-focused evaluation; low-latency demo.

## 2. Related Work
Prior phishing detection — lexical, classical ML, deep learning. Position this
work as interpretable + India-focused. ~10–15 citations.

## 3. Methodology
- **Data:** sources (PhishTank/OpenPhish, Tranco benign, UCI/Mendeley), sizes,
  class balance, collection date, fixed train/test split, no test-set tuning.
- **Features:** lexical; host-based (WHOIS/cert/ASN/redirects); brand-distance.
- **Models:** Logistic Regression, Random Forest, XGBoost.
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, latency.

## 4. Experiments & Results
Model comparison table; lexical vs. lexical+host ablation; ROC curves;
confusion matrices.

## 5. Interpretability Analysis
SHAP + permutation importance; which features drive detection; alignment with
security intuition.

## 6. India-Focused Evaluation
Curated India test set; metrics; structured error analysis (which scams slip
through and why). *(the novelty section)*

## 7. Discussion & Limitations
Evasion, dataset bias, feature drift, generalization.

## 8. Conclusion & Future Work
Established interpretable classical baseline; frame char-level CNN / transformer
URL encoder as the natural next step; browser-extension deployment; larger India
corpus; adversarial-evasion testing.

## References
*(BibTeX / IEEE references)*
