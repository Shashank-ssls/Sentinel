# Sentinel

**An interpretable, classical-ML detector for phishing / scam messages (smishing-first),
with a quantified study of how Western-trained models fail on India-specific scams —
and a small-data fix that closes the gap.**

Sentinel classifies an SMS **message** (not just a URL) as legitimate or unwanted/scam
using only **interpretable classical machine learning** (TF-IDF + structural lexical
features over scikit-learn models). Every verdict comes with a plain-language "why":
the exact terms and signals that drove it. A local, fully-offline web demo shows two
models side by side — a Western-only model and an India-augmented model — so you can
watch the generalization gap (and its fix) live.

> **No deep learning.** Neural approaches are documented as future work, never built.
> This keeps the model fully explainable end-to-end, which is the point of the study.

---

## Headline result

On a **sacred, held-out** set of real Indian SMS (never trained or TF-IDF-fit on):

| Model | Held-out India macro-F1 | Mendeley (Western) macro-F1 |
|-------|:----------------------:|:---------------------------:|
| **W** — Western-only | 0.768 | 0.987 |
| **A** — India-augmented | **0.991** (+0.223) | 0.986 (−0.001, no regression) |

A small India training sample closes the generalization gap **for free** — no cost to
Western performance. 104 India messages that W missed are caught by A, with the
explanation panel surfacing the India-specific vocabulary A learned (`airtel`, `rs`,
`click bit`, …) that W had effectively never seen.

---

## The research paper

This repository is the experimental backbone of an in-progress paper.

- **Working title:** *Interpretable Classical-ML Phishing-Link Detection with an
  India-Focused Evaluation*
- **Research question:** Can interpretable lexical + host-based features achieve
  phishing-detection performance competitive with heavier models while remaining fully
  explainable — and how well does such a model generalize to India-relevant scams
  (UPI, bank-impersonation, fake-KYC)?
- **Target venue:** student conference / IEEE workshop / IJRASET (TBD).
- **Contributions:** a reproducible de-duplicated 3-class SMS dataset; a classical-model
  comparison; a model-faithful interpretability analysis (global + per-message); a
  **quantified India generalization gap** with structured error analysis (the novelty);
  a small-data fix that closes it; and a low-latency, fully-offline, inspectable demo.

The full outline and all generated tables/figures live in [`paper/`](paper/)
(`paper/outline.md` + `paper/figures/`). Numbers in the paper are produced directly by
the scripts in this repo, in order — see **Reproduce the pipeline** below.

---

## Disclaimer

- **Research and educational use only.** Sentinel is a coursework / research prototype,
  **not** a production security control. Do **not** rely on it to keep you, your users,
  or your organization safe from real phishing or fraud.
- Predictions are **probabilistic and fallible**, biased toward the data they were
  trained on. False negatives (missed scams) and false positives (flagged legitimate
  messages) are expected.
- The included datasets are used for academic study only. Real SMS data may contain
  personal information; treat any message data responsibly and in line with the original
  dataset licenses and applicable privacy law. **No personal/secret data is committed**
  to this repository (raw data and trained models are git-ignored).
- Nothing here is legal, financial, or security advice.

---

## Quickstart (TL;DR)

Built and tested on **Python 3.12** (3.12.13).

```bash
git clone <your-repo-url> sentinel
cd sentinel

# create + activate a virtual environment
py -3.12 -m venv venv            # Windows (PowerShell): then  venv\Scripts\Activate.ps1
# python3.12 -m venv venv && source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
python -c "import src"           # sanity check
pytest -q                        # run the test suite
```

The trained models and raw datasets are **not** in the repo (git-ignored). To run the
demo you first regenerate them — see **Reproduce the pipeline**.

---

## Step-by-step setup (full)

This is the exact path to get an identical setup to the development machine.

**1. Prerequisites**
- Python **3.12** (`py -3.12 --version` on Windows, `python3.12 --version` elsewhere).
- Git.

**2. Clone and enter the project**
```bash
git clone <your-repo-url> sentinel
cd sentinel
```

**3. Virtual environment** (keeps dependencies isolated and reproducible)
```bash
# Windows (PowerShell)
py -3.12 -m venv venv
venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned

# macOS / Linux
python3.12 -m venv venv
source venv/bin/activate
```

**4. Install pinned dependencies** (versions are frozen for reproducibility)
```bash
pip install -r requirements.txt
```

**5. Configuration**
```bash
cp .env.example .env             # Windows: copy .env.example .env
```
`.env` holds non-secret config (`MODEL_PATH`, `CACHE_DB`, `RANDOM_SEED=42`). Defaults
are fine for the demo. `RANDOM_SEED=42` is what makes splits and training reproducible —
don't change it if you want byte-identical results.

**6. Provide the datasets** (not committed — place them under `data/raw/`)

The pipeline expects these files in `data/raw/`:

| File | What it is |
|------|------------|
| `Dataset_5971.csv` | Mendeley SMS corpus (ham/spam/smishing) — the Western training data |
| `india_real_sms.csv` | Real Indian SMS used for the India evaluation + augmentation |
| `india_synthetic_sms.csv` | Synthetic India 3-class probe (UPI/KYC/FASTag/Aadhaar patterns) |

Obtain the Mendeley SMS dataset from its original source and save it as
`data/raw/Dataset_5971.csv`. The India sets are described in
`Sentinel_Project_Brief.md` / `Sentinel_Smishing_Addendum.md`. (Only
`data/protected_brands.txt` and `data/sample_urls.csv` ship in the repo.)

**7. Verify**
```bash
python -c "import src"
pytest -q
```

---

## Reproduce the pipeline

Each phase is a single script, runnable top-to-bottom, in this order. Fixed seed and a
fixed train/test split make every run reproducible (nothing is tuned on the test set).
Run these to regenerate `data/processed/` and the `models/` artifacts the demo needs:

```bash
python -m src.data            # Phase 1 — build the de-duplicated 3-class train/test split
python -m src.train           # Phase 2 — feature pipeline + 4-model CV comparison, pick winner
python -m src.explain         # Phase 3 — production LogReg + global/local interpretability
python -m src.ablation        # Phase 4 — optional URL-feature ablation (honest negative result)
python -m src.eval_india      # Phase 5 — quantify the India generalization gap (eval only)
python -m src.india_augment   # Phase 5b — train the India-augmented model that closes the gap
```

Then make a one-off prediction from the command line:
```bash
python predict.py "Your KYC is suspended. Verify now: http://sbi-verify.example"
```

---

## Run the demo (local, fully offline — no network)

```bash
python -m uvicorn src.api:app --port 8000
# open http://127.0.0.1:8000 in a browser
```

`POST /analyze {text}` runs the message through **both** binary models (W and A) and
returns each verdict, confidence, and the top contributing features (reusing the same
`explain()` code path as training — no train/serve drift). Paste an Indian scam (or click
a preloaded chip) and watch **Model W** clear it (green "legitimate") while **Model A**
flags it (red "unwanted"), each with a "why" panel. The page also exposes the held-out
finding band (`/metrics`), per-message technical detail, an India-vocabulary callout, a
methodology panel, and a W-vs-A comparison table for inspection.

> Requires the artifacts from the steps above — if `models/` is empty, run the pipeline first.

---

## Repository layout

```
src/         feature extraction, models, API (ONE shared feature path for train + inference)
tests/       pytest suite (one test per src module)
data/        raw/ + processed/ datasets (git-ignored), protected_brands.txt, sample_urls.csv
models/      serialized model artifacts (git-ignored)
frontend/    threat/verdict UI served by the API
paper/       outline.md + figures/ (IEEE write-up; all tables/figures generated by src/)
notebooks/   exploration
```

Key docs: `Sentinel_Project_Brief.md` (what & why), `EXECUTION_PLAN.md` (ordered phase
plan), `Sentinel_Smishing_Addendum.md` (smishing-first framing that overrides the brief).

---

## Phase log (what each phase contributes)

- **Phase 0 — Scaffold & research setup.** Fresh venv installs from pinned
  `requirements.txt`; env-driven config; fixed seed; IEEE paper skeleton.
- **Phase 1 — Data pipeline (smishing-first).** Loads the Mendeley SMS corpus, normalizes
  to 3 lowercase classes (ham/spam/smishing), text-level dedup (5,971 → 5,949), then a
  fixed **stratified** train/test split (4759/1190, seed 42). Re-running reproduces a
  byte-identical split. *Paper:* the reproducible, de-duplicated dataset + class balance
  (~81% ham → macro-F1 headline).
- **Phase 2 — Features + model comparison.** Shared TF-IDF (1–2 gram) + structural feature
  pipeline; 4 classical models via stratified 5-fold CV on train only (TF-IDF fit inside
  each fold — no leakage); winner evaluated **once** on held-out test. RandomForest CV
  macro-F1 **0.884 ± 0.015**, held-out **0.889**. *Paper:* the first results table.
- **Phase 3 — Interpretability.** Retrains **LogisticRegression as the production model**
  (signed coefficients give honest explanations; within CV variance, test macro-F1
  **0.891**). Global top terms + signed structural weights (CSV + charts) and a
  permutation-importance cross-check; `explain(message)` returns per-message signed
  contributions. Top signals (claim/won/prize, has_phone, digit_count, has_url) match
  security intuition. *Paper:* the interpretability section.
- **Phase 4 — URL feature ablation (optional).** Adds 10 lexical URL features (no network).
  Stage-1 **0.8907** vs Stage-1+URL **0.8894** (Δ −0.0013) — URL features don't help
  overall (URLs appear in only ~3.8% of messages). *Paper:* an honest negative result;
  Stage-1 stays in production.
- **Phase 5 — India evaluation (eval-only).** Production model meets two India sets once.
  Real binary unwanted-vs-legit macro-F1 **0.792** vs Mendeley 0.982 (**gap −0.19**);
  synthetic 3-class **0.204** vs 0.891 (**gap −0.69**) — the Western model collapses to
  predicting `ham`, missing UPI/KYC/FASTag/Aadhaar smishing. *Paper:* the quantified
  India generalization gap with error analysis (the novelty).
- **Phase 5b — India-augmented model (the fix).** Real India set split 70/30 into a sacred
  held-out test. Two models, same pipeline: **W** (Mendeley only) and **A** (Mendeley +
  70% India). Held-out India macro-F1 **0.768 → 0.991 (+0.223)**; Mendeley **0.987 →
  0.986 (−0.001, no regression)**. *Paper:* the fix completing the novelty arc.
- **Phase 7 — Local demo (the finding, live).** Fully-offline FastAPI backend + single-page
  UI runs every message through both models and shows W-vs-A verdicts with "why" panels,
  a held-out finding band, India-vocabulary callout, methodology panel, and comparison
  table. *Paper:* the system/demo figure.

---

## Scope & future work

Classical ML only (scikit-learn, XGBoost). Documented as **future work**, not implemented:
neural-network URL encoders (char-level CNN / transformer), browser-extension deployment,
network host features (WHOIS/cert/ASN), a larger India corpus, and adversarial-evasion
testing.

---

## License

Released under the [MIT License](LICENSE).
