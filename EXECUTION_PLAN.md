# Sentinel — Claude Code Execution Plan & Folder Setup

> **Read order for the new chat / Claude Code:** read `Sentinel_Project_Brief.md` first (the what & why), then this file (the how & in what order). Start at Phase 0 only after the pre-flight files below exist.

---

## PART A — Files to place in the folder BEFORE Claude Code starts

Claude Code works far better when the skeleton, ground rules, and a tiny bit of seed data already exist. Create these by hand (or have a normal chat generate them) before kicking off the phased build. You do **not** need to write any real code — just scaffolding and rules.

### A1. Required (do these — they materially change output quality)

| File | Purpose | What goes in it |
|---|---|---|
| `Sentinel_Project_Brief.md` | the what & why | (already have it — drop it in) |
| `EXECUTION_PLAN.md` | the how & order | (this file) |
| `CLAUDE.md` | **standing rules for Claude Code** | project conventions it must follow every session — see template below. This is the single highest-value file. |
| `requirements.txt` | pin the stack | so env is reproducible — see template below |
| `.gitignore` | hygiene | ignore `venv/`, `__pycache__/`, `*.db`, `data/raw/`, model binaries, `.env` |
| `README.md` | front door | one-paragraph project description + the research question stated up top |
| `.env.example` | config contract | placeholder keys (e.g. `MODEL_PATH=`, optional `LLM_API_KEY=`) so secrets never get hard-coded |

### A2. Strongly recommended (small effort, big payoff)

| File / folder | Purpose |
|---|---|
| `data/sample_urls.csv` | 20-30 hand-picked rows (mix of known phishing + benign + a few India scam examples) with a `url,label` header. Lets Phase 2 run and be tested immediately without waiting on the full dataset download. |
| `data/protected_brands.txt` | one brand/domain per line (sbi, hdfcbank, icicibank, amazon, flipkart, paytm, phonepe, upi handles…). Powers the brand-distance feature. **You** are best placed to curate the India-relevant list. |
| `folder structure (empty dirs)` | create empty `data/raw/`, `data/processed/`, `models/`, `src/`, `notebooks/`, `tests/`, `frontend/`, `paper/` with `.gitkeep` files so Claude Code places things consistently. |

### A3. Optional (nice to have)

| File | Purpose |
|---|---|
| `paper/outline.md` | the IEEE section skeleton from the brief, so paper content has a home from day one |
| `Sentinel_Flowchart.png` | the architecture diagram, for the README and paper figure |
| a few real phishing/benign URLs you've personally seen | seeds your India set early (the slow task) |

> **You personally must own:** `data/protected_brands.txt` and the India examples in `sample_urls.csv`. Everything else Claude Code can generate. Your local knowledge of Indian scams is the part it can't fabricate well.

---

## PART B — `CLAUDE.md` template (paste this into the folder)

```markdown
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

## Out of scope (do not build unless asked)
- Deep learning, browser extension, user accounts/auth, cloud deployment,
  real-time streaming. These are future work.
```

---

## PART C — `requirements.txt` starter (paste into the folder)

```
# core ML
scikit-learn
xgboost
pandas
numpy
# interpretability
shap
# feature enrichment
tldextract
python-whois
python-Levenshtein
requests
# backend
fastapi
uvicorn[standard]
# utils / testing
python-dotenv
pytest
black
```
*(Frontend deps live separately under `frontend/` once Phase 7 starts.)*

---

## PART D — Phase-wise execution plan (for Claude Code)

> Tell Claude Code: "Do Phase N. Stop at the acceptance check and show me before continuing." Run phases **one at a time** — review each before the next. Don't let it sprint through all phases unsupervised; you need to understand each part for the viva.

### Phase 0 — Scaffold & research setup
**Build:** project tree (src/, tests/, data/, models/, notebooks/, frontend/, paper/), venv instructions, wire up `requirements.txt`, config loader reading `.env`, README with research question, empty `paper/outline.md` with IEEE headers.
**Acceptance:** `pip install -r requirements.txt` works in a fresh venv; repo tree matches conventions; `python -c "import src"` succeeds.
**Paper contribution:** none yet — sets reproducibility foundation.

### Phase 1 — Data pipeline
**Build:** `src/data.py` — loaders for PhishTank/OpenPhish + benign list, dedup, label normalization, a FIXED train/test split saved to `data/processed/` with a seed. Print dataset stats (counts, class balance, date). Validate against `data/sample_urls.csv` first.
**Acceptance:** running it produces versioned train/test CSVs + a printed stats summary; re-running gives identical split.
**Paper contribution:** Methodology §data; reproducibility guarantee.
**⚠️ Start curating the India set NOW in parallel — it's the slow task and your novelty.**

### Phase 2 — Lexical features + baseline models
**Build:** `src/features.py` (lexical only for now — length, dots, digits, @, IP-host, TLD, entropy, shortener, brand keywords). `src/train.py` trains LogReg + RandomForest + XGBoost, prints a comparison table (accuracy/precision/recall/F1/ROC-AUC) + confusion matrices, saves best model to `models/`.
**Acceptance:** one command outputs the metrics table; best model serialized; a CLI `predict.py <url>` returns a verdict.
**Paper contribution:** first real results table — this alone is a minimal viable paper.

### Phase 3 — Interpretability analysis
**Build:** `src/explain.py` — SHAP + permutation importance over the best model; save plots to `paper/figures/`; a function returning top-k contributing features for a single URL (feeds the app's "why" panel).
**Acceptance:** plots generated; `explain(url)` returns ranked human-readable reasons.
**Paper contribution:** Interpretability Analysis section + app explanation layer.

### Phase 4 — Host-based feature enrichment (OPTIONAL / time-permitting)
**Build:** extend `src/features.py` with WHOIS domain age, cert age, redirect count, ASN/country, brand-distance (Levenshtein to `protected_brands.txt`). Cache every lookup in SQLite. Retrain; report **lexical vs lexical+host** comparison.
**Acceptance:** second results table comparing feature sets; cached lookups make re-runs fast.
**Paper contribution:** second experiment (feature-set ablation).
**Note:** if time is tight, SKIP this before skipping Phase 5.

### Phase 5 — India-focused evaluation (PRIORITIZE — this is the novelty)
**Build:** `src/eval_india.py` — run the chosen model on the curated India test set; report metrics + a structured **error analysis** (which scam types slip through, why). Save a confusion matrix + a few qualitative examples.
**Acceptance:** India metrics table + written error analysis produced.
**Paper contribution:** the novelty section — your differentiator. Do not skip.

### Phase 6 — FastAPI service
**Build:** `src/api.py` — `POST /analyze {url}` → {verdict, confidence, top_features, enrichment}. Loads serialized model once at startup. SQLite cache. CORS enabled for the frontend.
**Acceptance:** `uvicorn` serves; curl returns a JSON verdict in <1s on a cached URL.
**Paper contribution:** the demo backend (System figure).

### Phase 7 — Threat-graph frontend
**Build:** `frontend/` — input box → animated force graph (URL→domain→redirects→ASN→cert→brand-distance), confidence dial, "why" panel. Calls the API. Record a 30s GIF.
**Acceptance:** live end-to-end demo from browser; three example URLs (phishing/benign/borderline) show distinct graphs.
**Paper contribution:** System figure + the faculty demo.
**Skill:** have the chat read `/mnt/skills/public/frontend-design/SKILL.md` first.

### Phase 8 — Paper write-up & packaging
**Build:** fill `paper/` sections with real numbers/tables/figures; honest Limitations; Dockerfile + docker-compose (api+frontend); GitHub Actions (pytest+black); README with demo GIF and run instructions.
**Acceptance:** paper draft complete; `docker-compose up` runs the whole stack; CI green.
**Paper contribution:** the submission.

---

## PART E — Minimum-viable path if time runs out

`Phase 0 → 1 → 2 → 3 → 5` = a complete, defensible paper + a CLI/simple demo.
Cut order if needed: drop **Phase 4 first**, then simplify **Phase 7** to a static visualization. **Never cut Phase 5** (the India evaluation is the novelty).

---

## PART F — Pre-flight checklist (tick before Phase 0)

- [ ] `Sentinel_Project_Brief.md` in folder
- [ ] `EXECUTION_PLAN.md` (this file) in folder
- [ ] `CLAUDE.md` written (Part B)
- [ ] `requirements.txt` written (Part C)
- [ ] `.gitignore` + `.env.example` created
- [ ] `README.md` with research question
- [ ] `data/sample_urls.csv` (20-30 seed rows incl. India examples)
- [ ] `data/protected_brands.txt` (India brands curated by you)
- [ ] empty dirs created (data/raw, data/processed, models, src, tests, frontend, paper) with .gitkeep
- [ ] venue + week-budget decided (sizes the phases)
- [ ] India-set curation STARTED (the slow task)

When every box above is ticked, open Claude Code and say:
**"Read Sentinel_Project_Brief.md and EXECUTION_PLAN.md, then do Phase 0. Stop at the acceptance check and show me before continuing."**
