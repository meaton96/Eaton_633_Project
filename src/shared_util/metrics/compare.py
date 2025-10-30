from __future__ import annotations

import math
import pathlib
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
DEFAULT_METRIC_COLS = ["roc_auc", "f1", "precision", "recall", "accuracy"]
TOP_K_FOR_RADAR = 5
FIGSIZE = (10, 6)
PALETTE = "viridis"  # change if you feel spicy

# ----------------------------
# Ranking logic
# ----------------------------
def rank_runs(
    df: pd.DataFrame,
    metric_cols=("roc_auc","f1","precision","recall","accuracy"),
    greater_is_better: dict[str, bool] | None = None,
    method: str = "ranksum",  # or "percentile_mean"
    weights: Optional[dict[str, float]] = None,  # per-metric weights
) -> pd.DataFrame:
    """
    method="ranksum": weighted sum of ranks across metrics, then min-max normalize to [0,1]
    method="percentile_mean": weighted average of per-metric rank percentiles (robust to ties/missing)
    """
    metric_cols = list(metric_cols)
    if greater_is_better is None:
        greater_is_better = {m: True for m in metric_cols}
    if weights is None:
        # default: everyone gets 1.0 unless you say otherwise
        weights = {m: 1.0 for m in metric_cols}

    out = df[["run_id","label"] + metric_cols].copy()

    # Rank each metric: best rank = 1
    for m in metric_cols:
        asc = not greater_is_better.get(m, True)
        if out[m].notna().any():
            out[f"rank_{m}"] = out[m].rank(ascending=asc, method="min")
        else:
            out[f"rank_{m}"] = np.nan

    rank_cols = [f"rank_{m}" for m in metric_cols]

    if method == "ranksum":
        # Weighted rank sum (ignore NaNs by filling with 0 weight contribution)
        w_rank_terms = []
        for m in metric_cols:
            w = float(weights.get(m, 1.0))
            col = out[f"rank_{m}"]
            w_rank_terms.append(w * col)

        out["rank_sum"] = pd.concat(w_rank_terms, axis=1).sum(axis=1, min_count=1)

        # Compute weighted best and worst possible sums given available columns
        best_sum = 0.0
        worst_sum = 0.0
        for m in metric_cols:
            w = float(weights.get(m, 1.0))
            col = out[f"rank_{m}"]
            if col.notna().any():
                best_sum += w * 1.0
                worst_sum += w * float(col.max())

        denom = float(worst_sum - best_sum)
        if denom <= 0 or not np.isfinite(denom):
            out["overall_score"] = 1.0
        else:
            out["overall_score"] = 1.0 - (out["rank_sum"] - best_sum) / denom

    elif method == "percentile_mean":
        # Convert ranks to "goodness" percentiles per metric, then weighted average
        pct_cols = []
        for m in metric_cols:
            rc = f"rank_{m}"
            col = out[rc]
            if col.notna().any():
                max_r = float(col.max())
                min_r = float(col.min())
                span = max(max_r - min_r, 1e-9)
                bad_pct = (col - min_r) / span
                good_pct = 1.0 - bad_pct
                out[rc + "_pct"] = good_pct
                pct_cols.append((rc + "_pct", float(weights.get(m, 1.0))))
            else:
                out[rc + "_pct"] = np.nan

        if pct_cols:
            cols, wts = zip(*pct_cols)
            wts = np.array(wts, dtype=float)
            wts = np.where(np.isfinite(wts), wts, 0.0)

            # Weighted mean across available metrics per row
            good = out[list(cols)]
            w_df = pd.DataFrame({c: w for c, w in zip(cols, wts)})
            num = (good * w_df).sum(axis=1, skipna=True)
            den = w_df.where(good.notna(), 0.0).sum(axis=1)
            out["overall_score"] = np.where(den > 0, num / den, np.nan)
            # if nothing available, set to NaN; sort will shove it down
        else:
            out["overall_score"] = np.nan
    else:
        raise ValueError("method must be 'ranksum' or 'percentile_mean'")

    # Sort: overall score desc, then precision, accuracy as tie-breakers, then f1, roc_auc
    return out.sort_values(
        ["overall_score", "precision", "accuracy", "f1", "roc_auc"],
        ascending=[False, False, False, False, False]
    ).reset_index(drop=True)



# ----------------------------
# Plotting helpers
# ----------------------------
def plot_metric_heatmap(df: pd.DataFrame, metric_cols: Iterable[str] = DEFAULT_METRIC_COLS, title: str = "Metrics per Run"):
    data = df[["label"] + list(metric_cols)].set_index("label")
    plt.figure(figsize=(max(8, len(metric_cols)*1.2), max(6, len(data)*0.4 + 2)))#type: ignore
    sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True)
    plt.title(title)
    plt.xlabel("Metric")
    plt.ylabel("Run (label)")
    plt.tight_layout()


def plot_bar_for_metric(df: pd.DataFrame, metric: str, title_prefix: str = ""):
    plt.figure(figsize=FIGSIZE)
    order = df.sort_values(metric, ascending=False)["label"]
    sns.barplot(data=df, x="label", y=metric, order=order, palette=PALETTE, hue=metric)
    plt.xticks(rotation=45, ha="right")
    ttl = f"{title_prefix}{metric}"
    plt.title(ttl)
    plt.xlabel("Run")
    plt.ylabel(metric)
    plt.tight_layout()


def plot_radar(
    df: pd.DataFrame,
    metric_cols: Iterable[str] = DEFAULT_METRIC_COLS,
    top_k: int = TOP_K_FOR_RADAR,
    title: str = "Radar: Top runs"
):
    metric_cols = list(metric_cols)
    # Normalize metrics to 0..1 for radar comparability
    metrics = df[metric_cols]
    norm = (metrics - metrics.min()) / (metrics.max() - metrics.min() + 1e-12)
    df_norm = pd.concat([df[["label"]], norm], axis=1)

    # Pick top_k by overall_score if present; else by f1
    if "overall_score" in df.columns:
        top = df_norm.loc[df["overall_score"].nlargest(top_k).index]
        labels_for_legend = df.loc[df["overall_score"].nlargest(top_k).index, "label"].tolist()
    else:
        top = df_norm.loc[df["f1"].nlargest(top_k).index]
        labels_for_legend = df.loc[df["f1"].nlargest(top_k).index, "label"].tolist()

    angles = np.linspace(0, 2*np.pi, len(metric_cols), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])  # close loop

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    for i, (_, row) in enumerate(top.iterrows()):
        vals = row[metric_cols].values
        vals = np.concatenate([vals, [vals[0]]])#type: ignore
        ax.plot(angles, vals, linewidth=2, label=labels_for_legend[i])
        ax.fill(angles, vals, alpha=0.1)

    ax.set_thetagrids(angles[:-1] * 180/np.pi, metric_cols)#type: ignore
    ax.set_title(title)
    ax.set_rlim(0, 1)#type: ignore
    ax.grid(True)
    plt.legend(bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()


# ----------------------------
# Leaderboards
# ----------------------------
def print_leaderboards(df_ranked: pd.DataFrame, metric_cols: Iterable[str] = DEFAULT_METRIC_COLS, top_n: int = 10):
    metric_cols = list(metric_cols)

    from IPython.display import display

    print("\n=== Overall ranking (by rank_sum -> overall_score) ===")
    cols = ["run_id","label","overall_score","rank_sum"] + metric_cols
    display(df_ranked[cols].head(top_n))

    for m in metric_cols:
        print(f"\n=== Top {top_n} by {m} ===")
        display(
            df_ranked.sort_values(m, ascending=False)[["run_id","label",m]]
            .head(top_n)
        )


# ----------------------------
# End-to-end entry point
# ----------------------------
def compare_runs(
    df: pd.DataFrame,
    metric_cols: Iterable[str] = DEFAULT_METRIC_COLS,
    title_suffix: str = "",
    weights: Optional[dict[str, float]] = None,  # per-metric weights
) -> pd.DataFrame:


    ranked = rank_runs(df, metric_cols=metric_cols, weights=weights)

    # Heatmap of metrics
    plot_metric_heatmap(df, metric_cols=metric_cols, title=f"Metrics per Run {title_suffix}".strip())

    # Bar charts for key metrics
    for m in metric_cols:
        plot_bar_for_metric(df, m, title_prefix="Runs by ")

    # Radar comparison for top K
    plot_radar(ranked, metric_cols=metric_cols, top_k=TOP_K_FOR_RADAR, title="Radar of top runs (normalized metrics)")

    # Print leaderboards
    print_leaderboards(ranked, metric_cols=metric_cols, top_n=min(10, len(ranked)))

    plt.show()
    return ranked
