"""Phase 2 — message-level feature extraction (Stage 1, classical, no NN).

ONE feature module used for BOTH training and live inference (no drift). It
produces a scikit-learn feature transformer combining two complementary sets:

1. **TF-IDF** over the message text (word-level, 1-2 grams). The vectorizer is
   part of the Pipeline, so it is fit *inside* each CV fold / on train only —
   never on the test set.
2. **Structural/lexical message features** (length, word/digit counts,
   uppercase ratio, link count, has_url/has_phone/has_email, currency-symbol
   count, urgency-keyword count).

There is deliberately NO URL lexical analysis here — that is Stage 2 (future).
``count_links`` and the inference-time flag derivation are simple presence
detection, not the Stage-2 lexical URL feature set.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import Levenshtein
import numpy as np
import pandas as pd
import tldextract
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

# Columns the feature pipeline consumes from an input DataFrame.
FEATURE_COLUMNS: list[str] = ["text", "has_url", "has_phone", "has_email"]

# Urgency / social-engineering keywords (kept in one place, lowercased).
URGENCY_KEYWORDS: tuple[str, ...] = (
    "verify",
    "blocked",
    "kyc",
    "win",
    "urgent",
    "otp",
    "free",
    "prize",
)

# Currency symbols counted as a structural signal.
CURRENCY_SYMBOLS: str = "$£€₹¥"

# Names of the structural features, in output-column order.
STRUCT_FEATURE_NAMES: list[str] = [
    "message_length",
    "word_count",
    "digit_count",
    "uppercase_ratio",
    "count_links",
    "has_url",
    "has_phone",
    "has_email",
    "count_currency",
    "urgency_count",
]

_URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:\+?\d[\s-]?){7,}")
_URGENCY_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in URGENCY_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def _structural_row(text: str, has_url: bool, has_phone: bool, has_email: bool):
    """Compute the structural feature vector for one message."""
    text = "" if text is None else str(text)
    length = len(text)
    word_count = len(text.split())
    digit_count = sum(ch.isdigit() for ch in text)
    alpha = [ch for ch in text if ch.isalpha()]
    uppercase_ratio = sum(ch.isupper() for ch in alpha) / len(alpha) if alpha else 0.0
    count_links = len(_URL_RE.findall(text))
    count_currency = sum(text.count(sym) for sym in CURRENCY_SYMBOLS)
    urgency_count = len(_URGENCY_RE.findall(text))
    return [
        float(length),
        float(word_count),
        float(digit_count),
        float(uppercase_ratio),
        float(count_links),
        float(bool(has_url)),
        float(bool(has_phone)),
        float(bool(has_email)),
        float(count_currency),
        float(urgency_count),
    ]


def compute_structural_features(df: pd.DataFrame) -> np.ndarray:
    """Vectorize structural features for a DataFrame into a dense float matrix.

    All values are non-negative (so the matrix is compatible with
    MultinomialNB downstream).
    """
    rows = [
        _structural_row(row["text"], row["has_url"], row["has_phone"], row["has_email"])
        for _, row in df.iterrows()
    ]
    return np.asarray(rows, dtype=float)


class StructuralFeatures(BaseEstimator, TransformerMixin):
    """Stateless transformer: DataFrame -> structural feature matrix."""

    def fit(self, X, y=None):  # noqa: D102 - stateless
        return self

    def transform(self, X) -> np.ndarray:  # noqa: D102
        return compute_structural_features(X)

    def get_feature_names_out(self, input_features=None):  # noqa: D102
        return np.asarray(STRUCT_FEATURE_NAMES, dtype=object)


# ---------------------------------------------------------------------------
# Stage 2 (Phase 4) — lexical URL features. Computed ONLY for messages that
# contain a URL; messages without one get an all-zero (neutral) vector so the
# feature space stays consistent. NO network calls (WHOIS/cert/ASN are future
# work) — purely lexical, fast, and reproducible.
# ---------------------------------------------------------------------------

URL_FEATURE_NAMES: list[str] = [
    "url_length",
    "url_count_dots",
    "url_count_digits",
    "url_has_at",
    "url_is_ip_host",
    "url_suspicious_tld",
    "url_subdomain_depth",
    "url_is_shortener",
    "url_is_https",
    "url_brand_similarity",
]

# Uncommon / frequently-abused TLDs (cheap suspicious-TLD signal).
SUSPICIOUS_TLDS: frozenset[str] = frozenset(
    {
        "xyz",
        "top",
        "tk",
        "ml",
        "ga",
        "cf",
        "gq",
        "online",
        "click",
        "link",
        "work",
        "loan",
        "win",
        "country",
        "kim",
        "party",
        "review",
        "stream",
        "gdn",
        "mom",
        "lol",
        "buzz",
        "rest",
        "fit",
        "zip",
        "mov",
        "icu",
        "cyou",
        "sbs",
        "shop",
    }
)

# Common URL shorteners (registrable domain).
URL_SHORTENERS: frozenset[str] = frozenset(
    {
        "bit.ly",
        "tinyurl.com",
        "goo.gl",
        "t.co",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "t.ly",
        "rb.gy",
        "cutt.ly",
        "shorturl.at",
        "rebrand.ly",
        "bitly.com",
        "tiny.cc",
        "tinyurl.co",
    }
)

_URL_EXTRACT_RE = re.compile(r"(?:https?://|www\.)[^\s]+", re.IGNORECASE)
_IP_HOST_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
# Offline extractor: bundled public-suffix snapshot, never fetches the network.
_TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

# Path to the brand list powering the lookalike (brand-distance) feature.
PROTECTED_BRANDS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "protected_brands.txt"
)


@lru_cache(maxsize=1)
def load_protected_brands(path: str | None = None) -> tuple[str, ...]:
    """Load brand tokens from ``protected_brands.txt`` (cached, lowercased)."""
    p = Path(path) if path else PROTECTED_BRANDS_PATH
    brands = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line:
            brands.append(line)
    return tuple(brands)


def _first_url(text: str) -> str | None:
    """Return the first URL-like token in the text, or None."""
    m = _URL_EXTRACT_RE.search(text or "")
    return m.group(0).rstrip(".,;:!?)\"'") if m else None


def _brand_similarity(domain: str) -> float:
    """Max normalized Levenshtein similarity (0..1) of domain to any brand.

    1.0 = exact match to a protected brand; 0.0 = no resemblance. Catches
    lookalikes like ``hdfc-bank-secure`` close to ``hdfcbank``.
    """
    domain = (domain or "").lower()
    if not domain:
        return 0.0
    best = 0.0
    for brand in load_protected_brands():
        m = max(len(domain), len(brand))
        if m == 0:
            continue
        sim = 1.0 - Levenshtein.distance(domain, brand) / m
        if sim > best:
            best = sim
    return best


def _url_row(text: str) -> list[float]:
    """Lexical URL feature vector for one message (zeros if no URL)."""
    url = _first_url(text)
    if not url:
        return [0.0] * len(URL_FEATURE_NAMES)
    is_https = 1.0 if url.lower().startswith("https://") else 0.0
    has_at = 1.0 if "@" in url else 0.0
    to_parse = url if re.match(r"^https?://", url, re.IGNORECASE) else "http://" + url
    host = (urlparse(to_parse).hostname or "").lower()
    is_ip = 1.0 if _IP_HOST_RE.match(host) else 0.0
    ext = _TLD_EXTRACT(to_parse)
    tld = ext.suffix.split(".")[-1] if ext.suffix else ""
    suspicious = 1.0 if tld in SUSPICIOUS_TLDS else 0.0
    subdomain_depth = float(len([p for p in ext.subdomain.split(".") if p]))
    reg_domain = ".".join(p for p in (ext.domain, ext.suffix) if p).lower()
    is_shortener = (
        1.0 if (reg_domain in URL_SHORTENERS or host in URL_SHORTENERS) else 0.0
    )
    return [
        float(len(url)),
        float(url.count(".")),
        float(sum(c.isdigit() for c in url)),
        has_at,
        is_ip,
        suspicious,
        subdomain_depth,
        is_shortener,
        is_https,
        _brand_similarity(ext.domain),
    ]


def compute_url_features(df: pd.DataFrame) -> np.ndarray:
    """Vectorize lexical URL features for a DataFrame (zeros for no-URL rows)."""
    rows = [_url_row(row["text"]) for _, row in df.iterrows()]
    return np.asarray(rows, dtype=float)


class URLFeatures(BaseEstimator, TransformerMixin):
    """Stateless transformer: DataFrame -> lexical URL feature matrix."""

    def fit(self, X, y=None):  # noqa: D102 - stateless
        return self

    def transform(self, X) -> np.ndarray:  # noqa: D102
        return compute_url_features(X)

    def get_feature_names_out(self, input_features=None):  # noqa: D102
        return np.asarray(URL_FEATURE_NAMES, dtype=object)


def build_feature_transformer(
    *,
    min_df: int = 2,
    ngram_range: tuple[int, int] = (1, 2),
    include_url: bool = False,
) -> Pipeline:
    """Build the combined feature transformer.

    Always includes TF-IDF + structural features (Stage 1). When
    ``include_url`` is True, the lexical URL block (Stage 2) is appended as an
    additional, combinable feature set. Returns a Pipeline (ColumnTransformer
    -> MaxAbsScaler); MaxAbsScaler preserves non-negativity and sparsity (so
    MultinomialNB still works) while putting wide-ranging features on a
    comparable scale to TF-IDF.
    """
    transformers = [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                min_df=min_df,
                ngram_range=ngram_range,
                sublinear_tf=True,
            ),
            "text",
        ),
        ("struct", StructuralFeatures(), FEATURE_COLUMNS),
    ]
    if include_url:
        transformers.append(("url", URLFeatures(), FEATURE_COLUMNS))
    column_transformer = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("columns", column_transformer), ("scale", MaxAbsScaler())])


def derive_flags(text: str) -> dict[str, bool]:
    """Derive has_url/has_phone/has_email from raw text (inference helper).

    Simple presence detection so live inference can build the same input row
    the trained pipeline expects. NOT the Stage-2 lexical URL features.
    """
    text = "" if text is None else str(text)
    return {
        "has_url": bool(_URL_RE.search(text)),
        "has_phone": bool(_PHONE_RE.search(text)),
        "has_email": bool(_EMAIL_RE.search(text)),
    }


def message_to_frame(text: str) -> pd.DataFrame:
    """Build a one-row input DataFrame (schema-compatible) from raw text."""
    flags = derive_flags(text)
    return pd.DataFrame([{"text": str(text), **flags}], columns=FEATURE_COLUMNS)
