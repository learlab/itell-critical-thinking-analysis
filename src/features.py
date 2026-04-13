"""Feature functions for summary revision analysis.

All feature functions operate on spaCy Doc objects (except Levenshtein, which
takes raw strings). Category-similarity features use en_core_web_lg word
vectors and Empath's seed word lists. Thresholds are data-driven: for each
category, we compute the within-list mean and SD of cosine similarity between
each seed word and its category centroid, then count essay tokens whose
similarity to the centroid exceeds mean - 1*SD.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import textdistance
from lexicalrichness import LexicalRichness

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

ALL_CATEGORIES = ["critical_thinking", "confusion"]


def load_seed_words() -> dict[str, list[str]]:
    """Load Empath's confusion seed words from the installed package's TSV,
    plus our custom critical_thinking seeds."""
    candidates = list(Path("/").glob("**/empath/data/categories.tsv"))
    if not candidates:
        raise FileNotFoundError("Cannot locate empath/data/categories.tsv")
    tsv_path = candidates[0]

    cats: dict[str, list[str]] = {}
    with open(tsv_path) as f:
        for line in f:
            cols = line.strip().split("\t")
            cats[cols[0]] = cols[1:]

    return {
        "critical_thinking": CRITICAL_THINKING_SEEDS,
        "confusion": cats["confusion"],
    }


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


def build_thresholds(
    nlp: spacy.Language,
    seeds: dict[str, list[str]],
    centroids: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute a data-driven similarity threshold for each category.

    For each seed list, we calculate the cosine similarity of every seed word
    to its own centroid, then set the threshold at mean - 1*SD. This means a
    token must be at least as similar to the centroid as the lower-end members
    of the category itself.
    """
    thresholds: dict[str, float] = {}
    for cat, words in seeds.items():
        vecs = [nlp.vocab[w].vector for w in words if nlp.vocab[w].has_vector]
        sims = np.array([_cosine_sim(v, centroids[cat]) for v in vecs])
        thresholds[cat] = float(sims.mean() - sims.std())
    return thresholds


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


def mtld(doc: Doc, threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).
    Uses the text from the spaCy Doc for consistency with other features."""
    text = doc.text.strip()
    if not text or word_count(doc) < 2:
        return 0.0
    return LexicalRichness(text).mtld(threshold=threshold)


# ---------------------------------------------------------------------------
# Category similarity features
# ---------------------------------------------------------------------------
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def category_similarity_counts(
    doc: Doc,
    centroids: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> dict[str, float]:
    """For each content token, compute cosine similarity to each category
    centroid and count tokens exceeding the data-driven threshold.
    Returns counts per 100 words."""
    wc = word_count(doc)
    if wc == 0:
        return {cat: 0.0 for cat in centroids}

    content = _content_tokens(doc)
    result: dict[str, float] = {}
    for cat, centroid in centroids.items():
        thresh = thresholds[cat]
        hits = sum(1 for tok in content if _cosine_sim(tok.vector, centroid) >= thresh)
        result[cat] = 100.0 * hits / wc
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
    thresholds: dict[str, float],
) -> dict[str, float]:
    """Compute the full feature set for a single summary text."""
    if not isinstance(text, str) or not text.strip():
        doc = nlp("")
    else:
        doc = nlp(text)

    feats: dict[str, float] = {
        "word_count": word_count(doc),
        "sentence_count": sentence_count(doc),
        "mtld": mtld(doc),
    }
    feats.update(category_similarity_counts(doc, centroids, thresholds))
    return feats


SURFACE_COLUMNS = [
    "word_count",
    "sentence_count",
    "mtld",
]

FEATURE_COLUMNS = SURFACE_COLUMNS + ALL_CATEGORIES
