# Sentinel — Phishing/Scam-Link Analyzer with a Live Threat Graph
### Project handoff & explanation brief (demo + research-paper edition)

> **How to use this file:** Paste this whole document into a new chat (or Claude Code) as the first message, then say which phase to start. It carries all context — builder background, research framing, methodology, phased plan, and paper structure — so nothing needs re-explaining.

---

## 1. Who I am & what I need from THIS project

- Final-year (7th sem) Data Science student. Comfortable: Python, pandas, scikit-learn, basic ML. Interested in cybersecurity (studying Security+), previously built a network traffic analyzer.
- **Two concrete goals, in priority order:**
  1. A **working demo I can show to faculty** — visual, live, immediately understandable.
  2. A **publishable research paper** (student conference / IEEE-style / a journal like IJraset, or a workshop) — so the project must be a *defensible experiment*, not just an app.
- **Hard constraints:**
  - **No CNN / no deep learning.** Use classical ML (logistic regression, random forest, gradient boosting / XGBoost). This is a deliberate choice, not a limitation — it's fast, explainable, and reviewer-friendly.
  - **Short on time.** Favor a tight scope that fully completes over an ambitious one that half-finishes. Every phase must leave me with something demoable.
  - Compute: GTX 1650 (4GB) + Kaggle/Colab free tiers. Classical ML trains fine locally.
- I keep deps off C: drive and use virtual environments.

---

## 2. The concept (one paragraph)

Sentinel takes any URL (or a forwarded scam SMS — highly relevant in India), extracts a set of **interpretable lexical and host-based features**, and classifies it as benign / suspicious / phishing using a **classical ML model**. The result is shown as an **animated threat graph**: the URL at the center, with nodes for its domain, redirects, hosting/ASN, certificate age, and lookalike-distance to known brands (SBI, HDFC, Amazon, UPI handles), edges lighting up red as risk accumulates. A **confidence dial** and a **plain-language "why"** (top contributing features via the model's own weights / feature importance) accompany every verdict.

The app is the demo. The **experiment behind it** — feature design, model comparison, interpretability analysis, error analysis on India-relevant scams — is the paper.

---

## 3. Research framing (this is what makes it publishable)

A paper needs a **question**, not just a build. Frame around one (or a combo) of these — all achievable with classical ML:

**Primary research question (pick one as the spine):**
- *"Can interpretable lexical + host-based features achieve phishing-detection performance competitive with heavier models, while remaining fully explainable?"* ← strongest fit for "no CNN" + interpretability angle.

**Supporting angles that add novelty (pick 1-2):**
- **India-specific evaluation.** Most phishing datasets are Western. Curate/augment a set of India-relevant scam URLs/SMS (UPI, bank-impersonation, fake-KYC, lottery) and report how a standard model performs on them vs. generic data. *This local angle is genuine novelty and easy to defend.*
- **Feature-importance / interpretability study.** Which features actually drive detection? Rank them (SHAP / permutation importance / model coefficients) and discuss what an analyst should look at. Explainability is a hot, publishable theme.
- **Lightweight & latency.** Report inference latency and model size — argue suitability for real-time / on-device / browser-extension deployment. Pairs naturally with "no deep learning."

**Honest contribution statement (what the paper claims):** a reproducible, interpretable, low-latency phishing detector with an India-focused evaluation and an analyst-facing visual explanation layer. That's a legitimate, modest, *publishable* contribution. Don't overclaim SOTA.

**On the neural network:** a char-level CNN / transformer URL encoder is **explicitly framed as future work, not built for this demo/paper.** The narrative is deliberate: establish a strong, interpretable classical baseline *first*, then position deep learning as the natural next step. This ordering is scientifically sound and reviewer-preferred — and it pre-positions a possible follow-up paper. Do not spend any time on the NN now beyond mentioning it in Future Work.

---

## 4. Data (decide early — this gates everything)

| Source | Use | Notes |
|---|---|---|
| **PhishTank** / **OpenPhish** | phishing URLs | free, widely used, citable. Check current access terms. |
| **Tranco list** / Alexa-style top sites | benign URLs | balanced negative class |
| **UCI / Mendeley phishing datasets** | ready-made features | good for a fast baseline + comparison to prior work |
| **India-relevant scrape/curation** | the novelty | collect scam SMS/URLs (your inbox, public scam-report forums, RBI/cybercrime advisories). Even a few hundred hand-labeled examples make a defensible India test set. |

> **Methodology rule for the paper:** fix a train/test split, keep a held-out test set untouched until the end, and never tune on it. Report dataset sizes, class balance, and collection date. Reproducibility = publishability.

---

## 5. Features (all interpretable, no DL)

**Lexical (from the URL string):** length, number of dots/hyphens/digits, presence of `@`, `//` after protocol, IP-as-host, suspicious TLD, subdomain depth, presence of brand keywords, entropy of the domain string, use of URL shorteners, `https` present.

**Host / network-based (optional, enrich if time):** domain age (WHOIS), certificate age/issuer, ASN / hosting country, number of redirects, reverse-DNS sanity.

**Lookalike / brand-distance:** Levenshtein / Jaro distance from the domain to a list of protected brands (banks, UPI, e-commerce) — catches `hdfc-bank-secure[.]xyz` style domains. This feature is intuitive in a demo *and* a nice paper feature.

> Start with lexical-only (needs no network calls, trains instantly, fully reproducible). Add host-based features as a second feature set so the paper can report "lexical vs. lexical+host" — that comparison is a free experimental result.

---

## 6. Models (classical, compared)

Train and **compare** at least three — the comparison table is a core paper artifact:
1. **Logistic Regression** — interpretable baseline, coefficients = explanation.
2. **Random Forest** — strong, gives feature importances.
3. **XGBoost / Gradient Boosting** — usually the top performer; pairs with SHAP.

Report per model: accuracy, precision, recall, **F1**, ROC-AUC, confusion matrix, and inference latency. Pick the best for the live demo. The **interpretability layer** (coefficients / feature importance / SHAP) feeds both the paper's analysis section and the app's "why" panel.

---

## 7. Tech stack (tight, demo-ready, free-tier)

| Layer | Choice | Why |
|---|---|---|
| ML | scikit-learn + XGBoost | classical, fast on 1650, explainable |
| Interpretability | SHAP + permutation importance | paper analysis + app "why" panel |
| Backend | FastAPI | clean API, matches my past work |
| Feature enrichment | `tldextract`, `python-whois`, `python-Levenshtein`, `requests` | lightweight |
| Frontend graph | D3.js **or** Cytoscape.js force graph | the animated threat graph = the wow |
| Storage | SQLite (cache lookups) | no heavy DB |
| Packaging | Docker + GitHub Actions | reproducibility for the paper |
| Paper | LaTeX (Overleaf, IEEE template) | standard, free |

---

## 7b. Efficiency principles (build lean — I'm short on time)

- **Lexical-only first, end to end.** Get data → features → model → result table → simple demo working on lexical features *alone* before touching any network-dependent feature (WHOIS/cert/ASN). Network calls are slow, flaky, and rate-limited; don't let them block the core loop.
- **One script per phase, runnable top-to-bottom.** Reproducibility = publishability, and it keeps me from re-running everything by hand.
- **Cache every external lookup** (SQLite) the first time, so the demo never waits on a live API and results are reproducible for the paper.
- **Freeze the test set on day one** and never touch it until final evaluation.
- **Don't gold-plate the frontend.** A clean force graph + dial + "why" panel is enough. Polish only after the paper's results exist.
- **Reuse, don't rebuild.** Prefer ready datasets (UCI/Mendeley) for the first baseline to get a results table fast; curate the India set in parallel.
- **No premature optimization, no NN.** If a step isn't in the minimum-viable-paper path (§9), it waits.

---

## 8. Phased plan (each phase leaves something demoable; paper grows alongside)

> After each phase, give me: (a) what's now demoable, and (b) a 2-3 sentence note on what this contributes to the paper.

**Phase 0 — Scaffold & research setup (½–1 day).** Repo, venv, requirements, README with the research question stated up front. Create an Overleaf doc from an IEEE conference template with section headers (Abstract, Intro, Related Work, Methodology, Experiments, Results, Discussion, Limitations, Conclusion). *Writing the skeleton now forces clear scope.*

**Phase 1 — Data pipeline (2–3 days).** Acquire PhishTank/OpenPhish + benign list; build a loader, dedup, label, fix a train/test split, save versioned CSVs. Report dataset stats. **Start curating the India test set in parallel** (this is the slow part — begin early).

**Phase 2 — Lexical features + baseline model (2–3 days).** Implement lexical feature extraction, train Logistic Regression + Random Forest + XGBoost, produce the **first results table** (accuracy/precision/recall/F1/AUC). *This alone is a minimal viable paper result and a CLI demo.*

**Phase 3 — Interpretability analysis (1–2 days).** SHAP + feature importance. Which features drive detection? Make the plots that go in the paper's analysis section. This also becomes the app's "why this verdict" panel.

**Phase 4 — Host-based feature enrichment (2–3 days, optional/time-permitting).** Add WHOIS domain age, cert age, redirects, ASN, brand-distance. Retrain; report **lexical vs. lexical+host** comparison — a second experimental result. Cache lookups in SQLite so the demo is fast and reproducible.

**Phase 5 — India-relevant evaluation (2–3 days).** Run the chosen model on the curated India test set. Report performance, do **error analysis** (which scams slip through and why). This is the paper's novelty section — prioritize it over Phase 4 if time is tight.

**Phase 6 — FastAPI service (2 days).** Wrap the model: `POST /analyze {url}` → verdict + confidence + top features + enrichment data. SQLite cache. This is the demo's backend.

**Phase 7 — The threat-graph frontend (3–4 days).** The visual centerpiece. Input box → animated force graph (URL → domain → redirects → ASN → cert → brand-distance), nodes coloring by risk, a confidence dial, and the "why" panel from Phase 3. Record a 30-second GIF for the README and the paper's "system" figure. **This is what you show faculty.**

**Phase 8 — Paper write-up & packaging (3–5 days).** Fill the Overleaf sections with real numbers, tables, and figures. Write Limitations honestly (dataset bias, evasion, feature staleness). Dockerize, add CI, finalize README with the demo GIF. Target a venue (student conference, IEEE workshop, or an open journal).

---

## 9. Minimum viable paper (if time gets very tight)

Phases 0 → 1 → 2 → 3 → 5 alone = a complete, publishable paper: *interpretable classical-ML phishing detection with an India-focused evaluation and feature-importance analysis.* Phases 4 and 7 are enhancements. **Cut host-features before you cut the India evaluation** — the local angle is your novelty. The demo (Phase 7) can be a simpler static visualization if needed; the paper doesn't require the animation, but faculty love it.

---

## 10. Demo script for faculty (90 seconds)

Paste a known phishing URL → graph animates, nodes flush red, confidence dial swings to 94% phishing, "why" panel lists *domain age 3 days, brand-distance 0.1 to 'hdfc', suspicious TLD*. Then paste a legit bank URL → graph stays green, low confidence. Then a borderline shortener → amber. Three clicks, the whole story lands.

---

## 11. Paper structure (IEEE-style)

1. **Abstract** — problem, approach, key result, India angle.
2. **Introduction** — phishing scale, India context, the interpretability gap, contributions (bulleted).
3. **Related Work** — prior phishing detection (lexical, ML, DL); position yours as interpretable + India-focused. ~10-15 citations.
4. **Methodology** — data, features (lexical, host, brand-distance), models, metrics.
5. **Experiments & Results** — model comparison table, lexical-vs-host, ROC curves, confusion matrices.
6. **Interpretability Analysis** — SHAP/feature importance, what drives detection.
7. **India-Focused Evaluation** — curated set, results, error analysis.
8. **Discussion & Limitations** — evasion, dataset bias, feature drift, generalization.
9. **Conclusion & Future Work** — frame the **neural network as the planned next phase**: argue the interpretable classical baseline is established here, and a char-level CNN / transformer URL encoder is the natural follow-up to capture sub-token patterns the lexical features miss. Also: real-time browser-extension deployment, larger India corpus, adversarial-evasion testing. *This gives the paper a forward-looking arc and pre-positions a potential second paper.*

---

## 12. Open questions to resolve in the next chat

- **Venue target?** (Decides format/rigor — student conf vs. IEEE workshop vs. open journal like IJRASET.) Pick early; it sets the bar.
- **India test set scope:** how many examples can I realistically curate, and from where? (Aim for a few hundred.)
- **Time budget in weeks** so phases can be sized to fit, and which phases to drop if needed.
- **SMS input in v1, or URLs only?** (SMS adds a text-parsing step; could be future work.)
- **Solo author or with a guide/co-author?** (Affects venue and framing.)

---

## 13. Interview/viva-defense checklist (also good for the paper's Q&A)

- Why classical ML over deep learning here? (interpretability, latency, data size, reproducibility)
- How did you prevent test-set leakage?
- What's the India set's novelty and how did you label it?
- What are the top features and does that match security intuition?
- How would an attacker evade this, and what's your honest limitation?

---

*End of brief. Next step: paste into a fresh chat, answer the §12 open questions, and start at Phase 0.*
