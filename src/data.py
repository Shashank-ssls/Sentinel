"""Phase 1 — data pipeline for the smishing-first design.

Loads the Mendeley SMS Phishing dataset (``data/raw/Dataset_5971.csv``),
normalizes it into the addendum schema, and produces a fixed, stratified
train/test split saved to ``data/processed/``.

Schema (one row per SMS message)::

    message_id  str   stable id, e.g. "mendeley-00042"
    text        str   the raw SMS text
    label       str   one of {ham, spam, smishing} (lowercased)
    has_url     bool  message contains a URL
    has_phone   bool  message contains a phone number
    has_email   bool  message contains an email address
    source      str   provenance, e.g. "mendeley" / "author-collected"
    india_relevant bool  India-curated evaluation flag

The schema deliberately leaves room for author-collected India rows to be
appended later (``source="author-collected"``, ``india_relevant=True``); that
ingestion is NOT built here, but nothing blocks it.

Run ``python -m src.data`` to (re)build the split and print dataset stats.
This module builds NO features and NO models (Phase 2) and NO URL-specific
logic (Stage 2) — by design.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROJECT_ROOT, get_config

# --- Paths -----------------------------------------------------------------
RAW_PATH: Path = PROJECT_ROOT / "data" / "raw" / "Dataset_5971.csv"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH: Path = PROCESSED_DIR / "train.csv"
TEST_PATH: Path = PROCESSED_DIR / "test.csv"

# --- Schema ----------------------------------------------------------------
SCHEMA: list[str] = [
    "message_id",
    "text",
    "label",
    "has_url",
    "has_phone",
    "has_email",
    "source",
    "india_relevant",
]
LABELS: tuple[str, ...] = ("ham", "spam", "smishing")

# Fraction of rows held out for the test set (stratified by label).
TEST_SIZE: float = 0.2


def _normalize_label(raw: str) -> str:
    """Lowercase a raw label and validate it is one of the 3 known classes.

    Handles the raw file's mixed casing (e.g. ``Smishing`` -> ``smishing``,
    ``Spam`` -> ``spam``).
    """
    label = str(raw).strip().lower()
    if label not in LABELS:
        raise ValueError(f"Unexpected label {raw!r} (normalized {label!r})")
    return label


def _yesno_to_bool(value: object) -> bool:
    """Convert a raw ``yes``/``No`` flag to a boolean (case-insensitive)."""
    token = str(value).strip().lower()
    if token in {"yes", "y", "true", "1"}:
        return True
    if token in {"no", "n", "false", "0"}:
        return False
    raise ValueError(f"Unexpected yes/no flag value: {value!r}")


def load_raw(raw_path: Path = RAW_PATH, dedup: bool = True) -> pd.DataFrame:
    """Load the raw Mendeley CSV and normalize it into the addendum schema.

    All rows are tagged ``source="mendeley"`` and ``india_relevant=False``.
    Row order follows the raw file so that ``message_id`` and the downstream
    split are reproducible.

    When ``dedup`` is True (the default), text-level deduplication is applied
    *before* any split: rows are deduplicated on ``text`` alone, keeping the
    first occurrence. This means an identical message can never span the
    train/test boundary, and a text appearing with conflicting labels collapses
    to its first label (identical messages can't carry two labels). Pass
    ``dedup=False`` to inspect the raw normalized rows (e.g. for stats).
    """
    raw = pd.read_csv(raw_path)
    expected = {"LABEL", "TEXT", "URL", "EMAIL", "PHONE"}
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(f"Raw file missing columns: {sorted(missing)}")

    df = pd.DataFrame(
        {
            "message_id": [f"mendeley-{i:05d}" for i in range(len(raw))],
            "text": raw["TEXT"].astype(str),
            "label": raw["LABEL"].map(_normalize_label),
            "has_url": raw["URL"].map(_yesno_to_bool),
            "has_phone": raw["PHONE"].map(_yesno_to_bool),
            "has_email": raw["EMAIL"].map(_yesno_to_bool),
            "source": "mendeley",
            "india_relevant": False,
        }
    )
    if dedup:
        df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    return df[SCHEMA]


def make_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a stratified (by ``label``) train/test split.

    The split is deterministic given a fixed ``seed`` and stable input order,
    so re-running reproduces the identical partition.
    """
    if seed is None:
        seed = get_config().random_seed
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build(
    test_size: float = TEST_SIZE,
    seed: int | None = None,
    raw_path: Path = RAW_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the split from the raw file and write it to ``data/processed/``."""
    df = load_raw(raw_path)
    train_df, test_df = make_split(df, test_size=test_size, seed=seed)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    return train_df, test_df


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the persisted train/test split as DataFrames (for Phase 2).

    Raises a helpful error if the split has not been built yet.
    """
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Processed split not found. Run `python -m src.data` first."
        )
    dtypes = {"has_url": bool, "has_phone": bool, "has_email": bool}
    train_df = pd.read_csv(TRAIN_PATH, dtype=dtypes)
    test_df = pd.read_csv(TEST_PATH, dtype=dtypes)
    return train_df, test_df


def _class_table(df: pd.DataFrame) -> str:
    """Render per-class counts and balance percentages for a split."""
    n = len(df)
    lines = []
    for label in LABELS:
        c = int((df["label"] == label).sum())
        pct = 100.0 * c / n if n else 0.0
        lines.append(f"    {label:<9} {c:>5}  ({pct:5.1f}%)")
    return "\n".join(lines)


def print_stats(
    df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """Print dataset stats: class balance, flag rates, and split check."""
    print("=" * 60)
    print(f"Sentinel - Phase 1 dataset stats  (source file: {RAW_PATH.name})")
    print("=" * 60)
    print(f"Total messages: {len(df)}")
    print("\nPer-class counts (full):")
    print(_class_table(df))

    print("\nFlag rates (full):")
    for flag in ("has_url", "has_phone", "has_email"):
        rate = 100.0 * df[flag].mean()
        print(f"    {flag:<10} {int(df[flag].sum()):>5}  ({rate:5.1f}%)")

    print(
        f"\nSplit: train={len(train_df)}  test={len(test_df)}  "
        f"(test_size={TEST_SIZE}, seed={get_config().random_seed})"
    )
    print("\nStratification check (per-class %; should match across splits):")
    print(f"    {'label':<9} {'full':>7} {'train':>7} {'test':>7}")
    for label in LABELS:
        f = 100.0 * (df["label"] == label).mean()
        tr = 100.0 * (train_df["label"] == label).mean()
        te = 100.0 * (test_df["label"] == label).mean()
        print(f"    {label:<9} {f:>6.1f}% {tr:>6.1f}% {te:>6.1f}%")
    max_drift = max(
        abs(
            100.0 * (train_df["label"] == lbl).mean()
            - 100.0 * (test_df["label"] == lbl).mean()
        )
        for lbl in LABELS
    )
    print(
        f"    -> max train/test class-share drift: {max_drift:.2f} pp "
        f"({'stratified OK' if max_drift < 1.0 else 'CHECK'})"
    )

    # Leakage guards: message_id AND text must be disjoint across splits.
    id_overlap = set(train_df["message_id"]) & set(test_df["message_id"])
    text_overlap = set(train_df["text"]) & set(test_df["text"])
    print(f"\nmessage_id overlap train&test: {len(id_overlap)} (must be 0)")
    print(f"text overlap train&test:       {len(text_overlap)} (must be 0)")
    print(f"remaining duplicate texts: {int(df['text'].duplicated().sum())}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Sentinel SMS split.")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    n_raw = len(load_raw(dedup=False))
    df = load_raw()
    print(
        f"Text-level dedup: raw {n_raw} -> {len(df)} unique messages "
        f"({n_raw - len(df)} duplicate texts dropped, first kept).\n"
    )
    train_df, test_df = build(test_size=args.test_size, seed=args.seed)
    print_stats(df, train_df, test_df)
    print(f"\nWrote:\n  {TRAIN_PATH}\n  {TEST_PATH}")


if __name__ == "__main__":
    main()
