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
**Phase 0 complete** — scaffold & reproducibility foundation in place. See
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

## Scope
Classical ML only (scikit-learn, XGBoost). Neural-network approaches are
documented as future work, not implemented in this version.
