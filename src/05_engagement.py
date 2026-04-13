"""Engagement stratification for STAIRS pairs.

Pulls BOTH sides of the STAIRS dialogue (user + bot turns) from the raw
chat_messages.csv.gz, computes per-pair engagement features, stratifies STAIRS
pairs into engagement tiers, and recomputes paired Cohen's d_z for the headline
features within each tier. Control (random_reread) stats are carried forward
from paired_stats.csv unchanged.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/home/jovyan/active-projects/itell-critical-thinking-analysis")
DATA = ROOT / "data"
RAW = Path("/home/jovyan/active-projects/itell-data/2024.05-CTTC/data/supabase_filtered")

HEADLINE_FEATURES = [
    "word_count", "sentence_count", "mtld",
    "critical_thinking", "confusion",
    "content_score", "language_score", "similarity_score", "containment_score",
]


def load_turns() -> pd.DataFrame:
    chat = pd.read_csv(RAW / "chat_messages.csv.gz")
    recs = []
    for row in chat.itertuples(index=False):
        try:
            turns = ast.literal_eval(row.data)
        except Exception:
            continue
        for tn in turns:
            if not isinstance(tn, dict):
                continue
            recs.append({
                "user_id": row.user_id,
                "page_slug": row.page_slug,
                "isUser": bool(tn.get("isUser", False)),
                "isStairs": bool(tn.get("isStairs", False)),
                "text": tn.get("text", "") or "",
                "ts_ms": tn.get("timestamp"),
            })
    t = pd.DataFrame(recs)
    t["ts"] = pd.to_datetime(t["ts_ms"], unit="ms", utc=True)
    return t


def engagement_for_pair(row, turns: pd.DataFrame) -> dict:
    sub = turns[
        (turns.user_id == row.user_id)
        & (turns.page_slug == row.page_slug)
        & (turns.isStairs)
        & (turns.ts >= row.created_at_t1)
        & (turns.ts <= row.created_at_t2)
    ]
    user_turns = sub[sub.isUser]
    bot_turns = sub[~sub.isUser]
    user_word_counts = user_turns.text.str.split().str.len().fillna(0).astype(int)
    if len(sub) >= 2:
        span = (sub.ts.max() - sub.ts.min()).total_seconds()
    else:
        span = 0.0
    return {
        "n_user_turns": int(len(user_turns)),
        "n_bot_turns": int(len(bot_turns)),
        "total_user_words": int(user_word_counts.sum()),
        "mean_user_turn_words": float(user_word_counts.mean()) if len(user_word_counts) else 0.0,
        "max_user_turn_words": int(user_word_counts.max()) if len(user_word_counts) else 0,
        "dialogue_span_seconds": float(span),
        "any_substantive_turn": bool((user_word_counts >= 10).any()),
    }


def cohens_dz(deltas: np.ndarray) -> float:
    sd = deltas.std(ddof=1)
    return float(deltas.mean() / sd) if sd > 0 and len(deltas) > 1 else 0.0


def paired_stats(deltas: np.ndarray) -> dict:
    n = len(deltas)
    dz = cohens_dz(deltas)
    if n < 2:
        return {"n": n, "mean_delta": float(deltas.mean()) if n else 0.0,
                "cohens_dz": dz, "p_t": np.nan, "p_wilcoxon": np.nan}
    t_stat, p_t = stats.ttest_1samp(deltas, 0.0)
    try:
        _, p_w = stats.wilcoxon(deltas, zero_method="wilcox")
    except ValueError:
        p_w = np.nan
    return {"n": n, "mean_delta": float(deltas.mean()), "cohens_dz": dz,
            "p_t": float(p_t), "p_wilcoxon": float(p_w)}


def main() -> None:
    pairs = pd.read_parquet(DATA / "revision_pairs_features.parquet")
    pairs["created_at_t1"] = pd.to_datetime(pairs["created_at_t1"], utc=True)
    pairs["created_at_t2"] = pd.to_datetime(pairs["created_at_t2"], utc=True)

    turns = load_turns()
    print(f"Total STAIRS turns loaded: {len(turns)}")

    stairs = pairs[pairs.condition == "stairs"].copy()
    eng = stairs.apply(lambda r: pd.Series(engagement_for_pair(r, turns)), axis=1)
    stairs = pd.concat([stairs.reset_index(drop=True), eng.reset_index(drop=True)], axis=1)

    # Describe
    print("\n=== Engagement distribution across 157 STAIRS pairs ===")
    print(stairs[["n_user_turns", "n_bot_turns", "total_user_words",
                  "mean_user_turn_words", "max_user_turn_words",
                  "dialogue_span_seconds"]].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2))

    # Tiers: 'none' = 0 user turns, 'low' = 1-2 user turns AND <10 total words,
    # 'medium' = anything between, 'high' = >=2 user turns AND >=20 total words.
    # Tertile split on total_user_words (equal-n bins).
    stairs["engagement_tier"] = pd.qcut(
        stairs.total_user_words, q=3,
        labels=["low", "medium", "high"], duplicates="drop",
    ).astype(str)
    print("\n=== Tier counts (tertiles of total_user_words) ===")
    print(stairs.engagement_tier.value_counts().reindex(["low", "medium", "high"]))
    print("\n=== total_user_words range per tier ===")
    print(stairs.groupby("engagement_tier").total_user_words.agg(["min", "max", "count"]))

    # Save engagement features
    eng_cols = ["user_id", "page_slug", "engagement_tier", "n_user_turns",
                "n_bot_turns", "total_user_words", "mean_user_turn_words",
                "max_user_turn_words", "dialogue_span_seconds",
                "any_substantive_turn"]
    stairs[eng_cols].to_parquet(DATA / "engagement_features.parquet", index=False)
    print(f"\nWrote {DATA / 'engagement_features.parquet'}")

    # Stratified paired stats
    control = pairs[pairs.condition == "random_reread"].copy()
    rows = []
    for feat in HEADLINE_FEATURES:
        delta_col = f"{feat}_delta"
        if delta_col not in pairs.columns:
            delta_col = f"{feat}_t2"
            if delta_col in pairs.columns:
                pairs[f"{feat}_delta"] = pairs[f"{feat}_t2"] - pairs[f"{feat}_t1"]
                stairs[f"{feat}_delta"] = stairs[f"{feat}_t2"] - stairs[f"{feat}_t1"]
                control[f"{feat}_delta"] = control[f"{feat}_t2"] - control[f"{feat}_t1"]
            else:
                continue
        # Control (full)
        ctrl_deltas = control[f"{feat}_delta"].dropna().to_numpy()
        rs = paired_stats(ctrl_deltas)
        rs.update({"feature": feat, "group": "control_full"})
        rows.append(rs)
        # Full STAIRS
        full_deltas = stairs[f"{feat}_delta"].dropna().to_numpy()
        rs = paired_stats(full_deltas)
        rs.update({"feature": feat, "group": "stairs_full"})
        rows.append(rs)
        # Per tier
        for tname in ["low", "medium", "high"]:
            sub = stairs[stairs.engagement_tier == tname]
            d = sub[f"{feat}_delta"].dropna().to_numpy()
            rs = paired_stats(d)
            rs.update({"feature": feat, "group": f"stairs_{tname}"})
            rows.append(rs)

    strat = pd.DataFrame(rows)[["feature", "group", "n", "mean_delta",
                                "cohens_dz", "p_t", "p_wilcoxon"]]
    strat.to_csv(DATA / "engagement_stratified_stats.csv", index=False)

    # Pretty wide view of Cohen's d_z
    pivot = strat.pivot(index="feature", columns="group", values="cohens_dz").round(3)
    col_order = ["control_full", "stairs_full", "stairs_low",
                 "stairs_medium", "stairs_high"]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    pivot = pivot.reindex(HEADLINE_FEATURES)
    print("\n=== Cohen's d_z by feature x group ===")
    print(pivot.to_string())

    # Engagement vs. key delta: continuous moderator
    print("\n=== Within-STAIRS: correlation of log(1+total_user_words) with feature delta ===")
    stairs["log_user_words"] = np.log1p(stairs.total_user_words)
    for feat in ["word_count", "sentence_count", "content_score",
                 "critical_thinking", "confusion"]:
        mask = stairs[f"{feat}_delta"].notna()
        r, p = stats.spearmanr(stairs.log_user_words[mask], stairs[f"{feat}_delta"][mask])
        print(f"  {feat:20s}  n={mask.sum():3d}  rho = {r:+.3f}  p = {p:.3f}")

    print(f"\nWrote {DATA / 'engagement_stratified_stats.csv'}")

    # -----------------------------------------------------------------
    # Figure: d_z by tier for the headline features
    # -----------------------------------------------------------------
    import matplotlib.pyplot as plt
    plot_feats = ["word_count", "sentence_count", "content_score",
                  "critical_thinking", "confusion"]
    feat_labels = {
        "word_count": "word count",
        "sentence_count": "sentence count",
        "content_score": "content",
        "critical_thinking": "cognition",
        "confusion": "uncertainty",
    }
    groups = ["control_full", "stairs_low", "stairs_medium", "stairs_high"]
    labels = ["control\n(n=110)",
              f"low\n(n={(stairs.engagement_tier=='low').sum()})",
              f"medium\n(n={(stairs.engagement_tier=='medium').sum()})",
              f"high\n(n={(stairs.engagement_tier=='high').sum()})"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(groups))
    width = 0.16
    for i, feat in enumerate(plot_feats):
        vals = [pivot.loc[feat, g] if g in pivot.columns else np.nan for g in groups]
        offset = (i - (len(plot_feats) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=feat_labels[feat])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Paired Cohen's $d_z$")
    ax.set_title("Effect sizes by STAIRS engagement tier")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    figpath = ROOT / "figs" / "engagement_tiers_dz.png"
    fig.savefig(figpath, dpi=150)
    print(f"Wrote {figpath}")


if __name__ == "__main__":
    main()
