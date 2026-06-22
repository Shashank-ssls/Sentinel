# Sentinel — Working Rules for Claude Code

## Project
Interpretable phishing/scam-link detector (classical ML, NO deep learning).
Goal: faculty demo + publishable paper. Short timeline — favor lean, finished
work over ambitious unfinished work.

## Hard rules
- NO neural networks / CNN / transformers in code. Classical ML only
  (scikit-learn, XGBoost). The NN is FUTURE WORK — mention only, never build.
- Every phase must end runnable and demoable. One script per phase, runnable
  top-to-bottom.
- Reproducibility first: fixed random seeds, fixed train/test split, versioned
  data files. Never tune on the test set.
- Cache all external lookups (WHOIS/cert/etc.) in SQLite. The live path must
  never block on a slow network call during a demo.
- Lexical features must work end-to-end BEFORE any network-dependent feature
  is added.
- Keep secrets in .env (never hard-code). Read config from environment.
- Use a venv. Pin versions in requirements.txt.
- Sentinel is SMS/smishing-FIRST. The unit of classification is the MESSAGE,
  not the URL. Read Sentinel_Smishing_Addendum.md — it overrides the brief.
- Headline metric is macro-F1, never plain accuracy (dataset is ~81% ham).

## Conventions
- Source in src/, tests in tests/, notebooks in notebooks/, model artifacts in
  models/, data in data/{raw,processed}/.
- Python: type hints, docstrings, black formatting, small functions.
- Each src module gets at least one pytest test.
- Same feature-extraction code path used for BOTH training and live inference
  (no drift). Put it in one module: src/features.py.

## Definition of done (per phase)
- Code runs top-to-bottom with one command.
- A short note added to README: what's now demoable + what it contributes to the paper.
- Tests pass.

## Workflow
- Do ONE phase at a time. Stop at the phase's acceptance check and show me
  before continuing. Do not sprint through multiple phases unsupervised.
- The ordered phase plan lives in EXECUTION_PLAN.md. The what & why lives in
  Sentinel_Project_Brief.md. Read both before starting Phase 0.

## Out of scope (do not build unless asked)
- Deep learning, browser extension, user accounts/auth, cloud deployment,
  real-time streaming. These are future work.
