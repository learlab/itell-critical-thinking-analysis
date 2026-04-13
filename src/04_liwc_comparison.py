"""Quick LIWC-vs-vector comparison for critical-thinking / confusion features.

Loads the LIWC dictionary in data/LIWC_DICT.txt, counts per-100-word hits for
Insight, CogMech, Tentat, and Discrep on t1/t2 summaries, then recomputes
paired Cohen's d_z and Wilcoxon stats to compare against the vector-similarity
headline results in paired_stats.csv.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/home/jovyan/active-projects/itell-critical-thinking-analysis")
DATA = ROOT / "data"

LIWC_CATS_OF_INTEREST = ["Insight", "CogMech", "Tentat", "Discrep"]
VECTOR_ANALOG = {
    "Insight": "critical_thinking",
    "CogMech": "critical_thinking",
    "Tentat": "confusion",
    "Discrep": "confusion",
}


def load_liwc(path: Path) -> dict[str, tuple[set[str], list[str]]]:
    """Return {category: (literal_words, prefix_patterns)}.

    File is CR-delimited (legacy Mac). Each line: category_name\tword\tword...
    Words ending in * are prefix wildcards.
    """
    with open(path, encoding="latin1", newline="") as f:
        text = f.read()
    cats: dict[str, tuple[set[str], list[str]]] = {}
    for line in text.split("\r"):
        toks = [t.strip().lower() for t in line.split("\t") if t.strip()]
        if not toks:
            continue
        name = toks[0]
        literals: set[str] = set()
        prefixes: list[str] = []
        for w in toks[1:]:
            if w.endswith("*"):
                prefixes.append(w[:-1])
            else:
                literals.add(w)
        # Preserve original capitalization of category name
        orig_name = line.split("\t", 1)[0].strip()
        cats[orig_name] = (literals, prefixes)
    return cats


TOKEN_RE = re.compile(r"[a-z']+")


def count_hits(text: str, literals: set[str], prefixes: list[str]) -> tuple[int, int]:
    """Return (n_category_hits, n_total_words) for one text."""
    if not isinstance(text, str) or not text.strip():
        return 0, 0
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0, 0
    hits = 0
    for t in tokens:
        if t in literals:
            hits += 1
            continue
        for p in prefixes:
            if t.startswith(p):
                hits += 1
                break
    return hits, len(tokens)


def per100(text: str, literals: set[str], prefixes: list[str]) -> float:
    hits, n = count_hits(text, literals, prefixes)
    return 0.0 if n == 0 else 100.0 * hits / n


def cohens_dz(deltas: np.ndarray) -> float:
    sd = deltas.std(ddof=1)
    return float(deltas.mean() / sd) if sd > 0 else 0.0


def main() -> None:
    pairs = pd.read_parquet(DATA / "revision_pairs_features.parquet")
    liwc = load_liwc(DATA / "LIWC_DICT.txt")

    print(f"Loaded {len(liwc)} LIWC categories; {len(pairs)} revision pairs\n")
    for cat in LIWC_CATS_OF_INTEREST:
        lits, pref = liwc[cat]
        print(f"  {cat}: {len(lits)} literals + {len(pref)} prefixes")
    print()

    for cat in LIWC_CATS_OF_INTEREST:
        lits, pref = liwc[cat]
        pairs[f"liwc_{cat}_t1"] = pairs["text_t1"].apply(lambda t: per100(t, lits, pref))
        pairs[f"liwc_{cat}_t2"] = pairs["text_t2"].apply(lambda t: per100(t, lits, pref))
        pairs[f"liwc_{cat}_delta"] = pairs[f"liwc_{cat}_t2"] - pairs[f"liwc_{cat}_t1"]

    # Load vector-based headline stats for side-by-side
    vec = pd.read_csv(DATA / "paired_stats.csv")
    vec = vec[vec.feature.isin(["critical_thinking", "confusion"])]

    rows = []
    for cat in LIWC_CATS_OF_INTEREST:
        col = f"liwc_{cat}_delta"
        for cond in ["stairs", "random_reread"]:
            sub = pairs[pairs.condition == cond][col].to_numpy()
            dz = cohens_dz(sub)
            t_stat, p_t = stats.ttest_1samp(sub, 0.0)
            try:
                w_stat, p_w = stats.wilcoxon(sub, zero_method="wilcox")
            except ValueError:
                w_stat, p_w = np.nan, np.nan
            rows.append({
                "liwc_category": cat,
                "vector_analog": VECTOR_ANALOG[cat],
                "condition": cond,
                "n": len(sub),
                "mean_delta": sub.mean(),
                "sd_delta": sub.std(ddof=1),
                "cohens_dz": dz,
                "t": t_stat,
                "p_t": p_t,
                "wilcoxon_W": w_stat,
                "p_wilcoxon": p_w,
            })
    liwc_stats = pd.DataFrame(rows)

    # Side-by-side: LIWC d_z vs vector d_z
    comp_rows = []
    for cat in LIWC_CATS_OF_INTEREST:
        analog = VECTOR_ANALOG[cat]
        for cond in ["stairs", "random_reread"]:
            liwc_dz = liwc_stats.query(
                "liwc_category == @cat and condition == @cond"
            )["cohens_dz"].iloc[0]
            vec_dz = vec.query(
                "feature == @analog and condition == @cond"
            )["cohens_dz"].iloc[0]
            comp_rows.append({
                "liwc_cat": cat,
                "vec_feature": analog,
                "condition": cond,
                "liwc_dz": round(liwc_dz, 3),
                "vec_dz": round(vec_dz, 3),
            })
    comp = pd.DataFrame(comp_rows)

    print("=== LIWC paired stats ===")
    print(liwc_stats.to_string(index=False))
    print("\n=== Side-by-side effect sizes (Cohen's d_z) ===")
    print(comp.to_string(index=False))

    out = DATA / "liwc_paired_stats.csv"
    liwc_stats.to_csv(out, index=False)
    comp.to_csv(DATA / "liwc_vs_vector_dz.csv", index=False)
    print(f"\nWrote {out}")
    print(f"Wrote {DATA / 'liwc_vs_vector_dz.csv'}")


if __name__ == "__main__":
    main()
