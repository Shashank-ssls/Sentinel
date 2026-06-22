# Sentinel

A live phishing/scam-link analyzer that classifies URLs (and scam SMS) using
**interpretable classical machine learning**, and visualizes the verdict as an
animated threat graph with a plain-language explanation.

## Research question
> Can interpretable lexical + host-based features achieve phishing-detection
> performance competitive with heavier models while remaining fully
> explainable — and how well does such a model generalize to India-relevant
> scams (UPI, bank-impersonation, fake-KYC)?

## Status
**Phase 1 complete** — smishing data pipeline in place (3-class SMS split). See
`EXECUTION_PLAN.md` for the phased plan and `Sentinel_Project_Brief.md` for full
context. Build proceeds one phase at a time.

## Quickstart
Built and tested on **Python 3.12**.
```bash
# Windows (PowerShell)
py -3.12 -m venv venv
venv\Scripts\activate
# macOS / Linux
python3.12 -m venv venv && source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env
python -c "import src"        # sanity check
pytest -q                     # run tests
```

## Project layout
```
src/         feature extraction, models, API (one shared feature path)
tests/       pytest suite (one test per src module)
data/        raw/ + processed/ datasets, sample_urls.csv, protected_brands.txt
models/      serialized model artifacts
notebooks/   exploration
frontend/    threat-graph UI (Phase 7)
paper/        outline.md + figures/  (IEEE write-up)
```

## Phase log
- **Phase 0 — Scaffold & research setup.** *Demoable:* fresh venv installs from
  pinned `requirements.txt`, `python -c "import src"` succeeds, `python -m
  src.config` prints resolved config, tests pass. *Paper contribution:* none yet
  — establishes the reproducibility foundation (fixed seed, env-driven config,
  pinned deps, IEEE paper skeleton).
- **Phase 1 — Data pipeline (smishing-first).** *Demoable:* `python -m src.data`
  loads the Mendeley SMS corpus (`data/raw/Dataset_5971.csv`), normalizes labels
  to 3 lowercase classes (ham/spam/smishing) and yes/No flags to booleans,
  emits the addendum schema, applies **text-level deduplication** (raw 5,971 →
  5,949 unique messages; dedup on `text`, first kept, before the split so no
  text spans train/test), then writes a fixed **stratified** train/test split
  (4759/1190, seed 42) to `data/processed/`; re-running reproduces a
  byte-identical split. Prints class balance, flag rates, stratification check,
  and message_id + text leakage guards. `load_processed()` serves Phase 2.
  *Paper contribution:* Methodology §data — reproducible, de-duplicated 3-class
  SMS dataset, class balance (~81% ham → macro-F1 headline), and the schema that
  makes the India evaluation subset (Phase 5) a simple `india_relevant` filter.
- **Phase 2 — Features + model comparison (Stage 1).** *Demoable:* `python -m
  src.train` builds the shared TF-IDF (1–2 gram) + structural feature pipeline
  (`src/features.py`, one code path for train and inference), compares 4
  classical models via stratified 5-fold CV on train only (no leakage — TF-IDF
  fit inside each fold), then evaluates the winner **once** on the held-out
  test set. `python predict.py "<msg>"` returns class + probabilities. Best
  model: **RandomForest**, CV macro-F1 **0.884 ± 0.015**, held-out test
  macro-F1 **0.889** (binary smishing-vs-not 0.926). Artifacts saved under
  `models/` (`cv_comparison.csv`, `test_results.json`, `confusion_matrix.csv`,
  `best_model.pkl`). *Paper contribution:* the first results table — model
  comparison + held-out per-class/confusion metrics, a minimal viable paper
  result. (Note: on out-of-distribution India-style messages the model is
  uncertain — motivates the Phase 5 India evaluation.)
- **Phase 3 — Interpretability (Stage 1).** *Demoable:* `python -m src.explain`
  retrains **LogisticRegression as the production model** (chosen over RF: within
  CV variance, test macro-F1 **0.891**, but signed coefficients give honest
  explanations; RF artifacts kept), then writes global + local interpretability
  artifacts. *Global:* top positive/negative TF-IDF terms per class and signed
  structural-feature weights (CSV + bar charts in `paper/figures/`), plus a
  permutation-importance cross-check. *Local:* `explain(message)` returns the
  predicted class, probabilities, and top signed feature contributions (the
  demo's future "why" panel). Top smishing terms (claim/won/prize/http) and
  structural signals (has_phone, digit_count, has_url) match security intuition;
  permutation importance agrees (digit_count, has_phone, has_url top).
  *Paper contribution:* the Interpretability Analysis section — what drives
  detection, with model-faithful global and per-message explanations.
- **Phase 4 — Stage 2: URL feature ablation (optional).** *Demoable:* `python -m
  src.ablation` extends `src/features.py` with 10 lexical URL features
  (length, dots, digits, `@`, IP-host, suspicious TLD, subdomain depth,
  shortener, https, brand-distance via Levenshtein to `protected_brands.txt`),
  computed only for URL-bearing messages (neutral zeros otherwise), no network
  calls. Same LogReg/seed/split/5-fold CV. **Ablation result:** Stage-1 test
  macro-F1 **0.8907** vs Stage-1+URL **0.8894** (Δ −0.0013; CV Δ −0.0023) — URL
  features **do not help overall**, as expected since URLs appear in only ~3.8%
  of messages. `explain()` now surfaces `url__*` contributions on URL-bearing
  messages. Ablation table saved to `paper/figures/url_ablation.csv`; augmented
  model saved separately (`models/url_augmented_model.pkl`) — **Stage-1 remains
  the production model**; all prior artifacts untouched. *Paper contribution:*
  the feature-set ablation (Stage-1 vs Stage-1+URL) — an honest negative result
  showing message-level features already capture the signal. Network host
  features (WHOIS/cert/ASN) remain future work.
- **Phase 5 — India evaluation (hybrid, evaluation-only).** *Demoable:* `python
  -m src.eval_india` loads the production model (no retraining; training split
  untouched) and meets two India sets once. *Dataset A (real Indian SMS,
  binary):* cleaned 2267 → 2061 (−1 null, −27 "image omitted", −178 dup);
  real-world **unwanted-vs-legit** binary macro-F1 **0.792** vs Mendeley's
  comparable 0.982 (**gap −0.19**) — high precision (0.93) but low recall (0.57)
  on unwanted; misses mostly Airtel/recharge promos. *Dataset B (synthetic
  India 3-class probe):* 3-class macro-F1 **0.204** vs Mendeley 0.891 (**gap
  −0.69**) — the Western-trained model collapses to predicting `ham`, catching
  almost no UPI/KYC/FASTag/Aadhaar smishing. Honest framing: A is a real-world
  binary claim, B a controlled pattern probe (not a real-world claim). Artifacts
  + error analysis + misclassification examples saved to `paper/figures/`.
  *Paper contribution:* the novelty section — a quantified India generalization
  gap with error analysis, showing a model trained on Western smishing fails to
  transfer to India-specific scam patterns.

## Scope
Classical ML only (scikit-learn, XGBoost). Neural-network approaches are
documented as future work, not implemented in this version.
