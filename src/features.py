"""Feature functions for summary revision analysis.

All feature functions operate on spaCy Doc objects (except Levenshtein, which
takes raw strings). Category-similarity features use en_core_web_lg word
vectors and Empath's seed word lists, replicating Empath's approach locally.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import textdistance

if TYPE_CHECKING:
    import spacy
    from spacy.tokens import Doc

# ---------------------------------------------------------------------------
# Seed words
# ---------------------------------------------------------------------------
CRITICAL_THINKING_SEEDS = [
    "analyze", "compare", "contrast", "evaluate", "infer", "explain",
    "justify", "interpret", "consider", "assume", "conclude",
]

EMPATH_BUILTIN_CATEGORIES = [
    "science", "philosophy", "negotiate", "communication", "confusion",
]

ALL_CATEGORIES = EMPATH_BUILTIN_CATEGORIES + ["critical_thinking"]

# Similarity bins: token cosine-similarity to category centroid.
# Calibrated against en_core_web_lg (300-dim GloVe). Seed-word self-similarity
# medians range 0.52–0.76; cross-category means 0.17–0.36.
SIMILARITY_BINS = [
    ("strong",   0.50, np.inf),   # core category words
    ("moderate", 0.35, 0.50),     # domain-adjacent
    ("weak",     0.20, 0.35),     # marginal association
]


def load_seed_words() -> dict[str, list[str]]:
    """Load Empath built-in seed words from the installed package's TSV,
    plus our custom critical_thinking seeds."""
    # Find the empath categories.tsv — try common install locations.
    candidates = list(Path("/").glob("**/empath/data/categories.tsv"))
    if not candidates:
        raise FileNotFoundError("Cannot locate empath/data/categories.tsv")
    tsv_path = candidates[0]

    cats: dict[str, list[str]] = {}
    with open(tsv_path) as f:
        for line in f:
            cols = line.strip().split("\t")
            cats[cols[0]] = cols[1:]

    seeds = {c: cats[c] for c in EMPATH_BUILTIN_CATEGORIES}
    seeds["critical_thinking"] = CRITICAL_THINKING_SEEDS
    return seeds


def build_centroids(
    nlp: spacy.Language, seeds: dict[str, list[str]]
) -> dict[str, np.ndarray]:
    """Compute the mean vector (centroid) for each category's seed list."""
    centroids: dict[str, np.ndarray] = {}
    for cat, words in seeds.items():
        vecs = [nlp.vocab[w].vector for w in words if nlp.vocab[w].has_vector]
        if not vecs:
            raise ValueError(f"No vectors found for category '{cat}'")
        centroids[cat] = np.mean(vecs, axis=0)
    return centroids


# ---------------------------------------------------------------------------
# spaCy-based surface features
# ---------------------------------------------------------------------------
def _content_tokens(doc: Doc) -> list:
    """Alphabetic, non-stop tokens with vectors — used for similarity scoring."""
    return [t for t in doc if t.is_alpha and t.has_vector and not t.is_stop]


def word_count(doc: Doc) -> int:
    """Count of alphabetic tokens (including stop words)."""
    return sum(1 for t in doc if t.is_alpha)


def sentence_count(doc: Doc) -> int:
    sents = list(doc.sents)
    return max(1, len(sents)) if doc.text.strip() else 0


def type_token_ratio(doc: Doc) -> float:
    toks = [t.lower_ for t in doc if t.is_alpha]
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


# ---------------------------------------------------------------------------
# Category similarity features
# ---------------------------------------------------------------------------
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def category_similarity_bins(
    doc: Doc, centroids: dict[str, np.ndarray]
) -> dict[str, float]:
    """For each content token, compute cosine similarity to each category
    centroid and bin into tiers. Returns counts per 100 words."""
    wc = word_count(doc)
    if wc == 0:
        return {
            f"{cat}_{bname}": 0.0
            for cat in centroids
            for bname, _, _ in SIMILARITY_BINS
        }

    # Pre-compute similarities: (n_tokens, n_categories)
    content = _content_tokens(doc)
    result: dict[str, float] = {}
    for cat, centroid in centroids.items():
        bin_counts = {bname: 0 for bname, _, _ in SIMILARITY_BINS}
        for tok in content:
            sim = _cosine_sim(tok.vector, centroid)
            for bname, lo, hi in SIMILARITY_BINS:
                if lo <= sim < hi:
                    bin_counts[bname] += 1
                    break
        for bname in bin_counts:
            result[f"{cat}_{bname}"] = 100.0 * bin_counts[bname] / wc
    return result


# ---------------------------------------------------------------------------
# Pairwise features
# ---------------------------------------------------------------------------
def levenshtein_distance(t1: str, t2: str) -> int:
    a = t1 if isinstance(t1, str) else ""
    b = t2 if isinstance(t2, str) else ""
    return textdistance.levenshtein.distance(a, b)


def normalized_levenshtein(t1: str, t2: str) -> float:
    a = t1 if isinstance(t1, str) else ""
    b = t2 if isinstance(t2, str) else ""
    denom = max(len(a), len(b))
    if denom == 0:
        return 0.0
    return textdistance.levenshtein.distance(a, b) / denom


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def compute_all(
    text: str,
    nlp: spacy.Language,
    centroids: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute the full feature set for a single summary text."""
    if not isinstance(text, str) or not text.strip():
        doc = nlp("")
    else:
        doc = nlp(text)

    feats: dict[str, float] = {
        "word_count": word_count(doc),
        "sentence_count": sentence_count(doc),
        "type_token_ratio": type_token_ratio(doc),
    }
    feats.update(category_similarity_bins(doc, centroids))
    return feats


SURFACE_COLUMNS = [
    "word_count",
    "sentence_count",
    "type_token_ratio",
]

CATEGORY_BIN_COLUMNS = [
    f"{cat}_{bname}"
    for cat in ALL_CATEGORIES
    for bname, _, _ in SIMILARITY_BINS
]

FEATURE_COLUMNS = SURFACE_COLUMNS + CATEGORY_BIN_COLUMNS
