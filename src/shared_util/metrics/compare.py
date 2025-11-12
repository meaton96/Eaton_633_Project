from __future__ import annotations

import math
import pathlib
from typing import Iterable, Optional


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from shared_util.metrics.printing import print_leaderboards
from shared_util.metrics.plotting import DEFAULT_METRIC_COLS, plot_bar_for_metric, plot_metric_heatmap


def rank_runs(
    df: pd.DataFrame,
    metric_cols=("roc_auc", "f1", "precision", "recall", "accuracy"),
    greater_is_better: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """
    Ranks each run by averaging ranks across metrics (no weights).
    """
    metric_cols = list(metric_cols)
    if greater_is_better is None:
        greater_is_better = {m: True for m in metric_cols}

    out = df[["run_id", "label"] + metric_cols].copy()

    # Rank each metric: best rank = 1
    for m in metric_cols:
        asc = not greater_is_better.get(m, True)
        if out[m].notna().any():
            out[f"rank_{m}"] = out[m].rank(ascending=asc, method="min")
        else:
            out[f"rank_{m}"] = np.nan

    rank_cols = [f"rank_{m}" for m in metric_cols]

    # Unweighted rank sum (ignores NaNs)
    out["rank_sum"] = out[rank_cols].sum(axis=1, min_count=1)

    # Compute best and worst possible sums given available ranks
    best_sum = len(metric_cols) * 1.0
    worst_sum = sum(out[rank_cols].max(skipna=True))

    denom = float(worst_sum - best_sum)
    if denom <= 0 or not np.isfinite(denom):
        out["overall_score"] = 1.0
    else:
        out["overall_score"] = 1.0 - (out["rank_sum"] - best_sum) / denom

    # Sort: overall score desc, then precision, accuracy as tie-breakers
    return out.sort_values(
        ["overall_score", "precision", "accuracy", "f1", "roc_auc"],
        ascending=[False, False, False, False, False]
    ).reset_index(drop=True)



# public api

def print_run_compare(df: pd.DataFrame,
    metric_cols: Iterable[str] = DEFAULT_METRIC_COLS):

    ranked = rank_runs(df, metric_cols=metric_cols)

    # Print leaderboards
    print_leaderboards(ranked, metric_cols=metric_cols, top_n=min(10, len(ranked)))


# public entry point
def plot_compare(
    df: pd.DataFrame,
    metric_cols: Iterable[str] = DEFAULT_METRIC_COLS,
    title_suffix: str = "",
    h_cmap='YlGnBu',
    figsize=(10,6),
    base_palette='viridis'
):


    
    # Heatmap of metrics
    plot_metric_heatmap(
        df, 
        metric_cols=metric_cols, 
        title=f"Metrics per Run {title_suffix}".strip(),
        cmap=h_cmap
    )

    def make_label_palette(df: pd.DataFrame, base_palette: str = base_palette):
        import seaborn as sns
        labels = pd.Index(df["label"].astype(str).unique())
        colors = sns.color_palette(base_palette, n_colors=len(labels))
        return dict(zip(labels, colors))

    label_palette = make_label_palette(df, base_palette=base_palette)  

    for m in metric_cols:
        plot_bar_for_metric(
            df,
            m,
            title_prefix="Runs by ",
            figsize=figsize,
            palette=label_palette,
        )


    plt.show()
