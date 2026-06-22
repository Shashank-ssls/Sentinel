# Sentinel — Smishing-First Addendum (READ THIS — it overrides parts of the brief)

> The original brief/flowchart framed Sentinel as a **URL classifier**. We have
> pivoted to **smishing-first**: the unit of classification is the **SMS message**,
> not the URL. Where this addendum conflicts with the brief, **this wins**. The
> URL work is preserved as stage two (see below), so nothing is wasted.

## Locked design decisions

1. **Classification target: 3-class — ham / spam / smishing.**
   - Headline metric is **macro-F1** (NOT plain accuracy — the set is ~81% ham,
     so accuracy is misleading). Report per-class precision/recall + confusion
     matrix. May ALSO report a collapsed binary (smishing vs not) as a free extra.
   - Handle class imbalance explicitly: stratified split + class weights.

2. **Two-stage architecture, built IN THIS ORDER (not interleaved):**
   - **Stage 1 (build first, completely): message classifier.** Classifies the
     SMS text into ham/spam/smishing. This alone is a complete paper.
   - **Stage 2 (enhancement, only after Stage 1 is done): URL analysis.** If a
     message contains a URL, run the lexical URL features from the original brief
     on it. If time runs short, Stage 2 degrades gracefully to "future work."

3. **India angle: train on general smishing, evaluate on India.**
   - **Training:** Mendeley SMS Phishing Dataset (Mishra & Soni, 2022,
     DOI 10.17632/f45bkkt8pr.1) — `data/raw/Dataset_5971.csv`: a general /
     largely Western-origin smishing corpus of ~5,971 messages
     (4,844 ham / 489 spam / 638 smishing), columns LABEL/TEXT/URL/EMAIL/PHONE.
     (Inspection of the raw data confirms it is largely Western/UK-origin —
     BankOfAmerica, Orange, £ amounts — NOT India-origin.)
   - **Evaluation:** my own curated India SMS set (inbox + chakshu/cybercrime
     reports), tagged India-relevant. This is the generalization test, and the
     gap between general-smishing training and India evaluation IS the novelty —
     start collecting now.

4. **Demo visualization: deferred.** Build the model first; decide viz later.
   (Likely: highlight suspicious tokens + risk dial + "why" panel. Not a URL graph.)

## Revised features (message-level, all classical — NO neural nets)

- **Text features:** TF-IDF over message tokens (classical, interpretable —
  feature weights ARE the explanation). This is the core of Stage 1.
- **Structural/lexical message features:** message length, word count, digit
  count, uppercase ratio, count of links, presence-of-URL / presence-of-phone /
  presence-of-email flags (Mendeley already provides these), count of currency
  symbols, urgency/keyword signals (verify, blocked, KYC, win, urgent, OTP).
- **Stage 2 only (when URL present):** the original lexical URL features
  (length, dots, @, IP-host, TLD, entropy, shortener, brand-distance to
  protected_brands.txt).

> Keep one feature module (`src/features.py`) used by BOTH training and
> inference — no drift. TF-IDF vectorizer must be fit on train only and saved
> alongside the model.

## Models (classical, compared)

Multinomial Naive Bayes (strong TF-IDF baseline), Logistic Regression, Random
Forest, XGBoost. Compare on macro-F1. Naive Bayes is a known strong/ fast
baseline for SMS text — include it.

## Revised data schema (Phase 1 must build this)

Each row: `message_id, text, label (ham/spam/smishing), has_url, has_phone,
has_email, source (mendeley/author-collected/...), india_relevant (bool)`.
The `source` + `india_relevant` columns make the India evaluation subset a
simple filter in Phase 5, not a retrofitted pipeline.

> **Normalization note (Phase 1):** labels in `data/raw/Dataset_5971.csv` are
> mixed-case (`ham`, `spam`, `Smishing`) and must be lowercased to exactly 3
> classes; the `yes`/`No` values in the URL/EMAIL/PHONE columns must be
> converted to booleans.

## License note (verify before publishing)

Confirm the Mendeley dataset license permits academic use/redistribution
(check the DOI page's license line). Cite it correctly regardless.
