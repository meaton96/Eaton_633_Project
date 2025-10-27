import pandas as pd
import numpy as np


def _filter_duplicates_by_id(df, keep_id):
    
    key_cols = ['model', 'data', 'threshold_notes', 'pipeline_notes']
    
    
    duplicated_mask = df.duplicated(subset=key_cols, keep=False)

    duplicates = df[duplicated_mask]
    uniques = df[~duplicated_mask]
    
    kept_duplicates = duplicates[duplicates['id'] == keep_id]
    
    result = pd.concat([uniques, kept_duplicates], ignore_index=True)
    
    return result.sort_values(by='roc_auc', ascending=False).reset_index(drop=True)



def plot_metrics(df, plot_id, metrics = ["roc_auc", "accuracy", "precision", "recall", "f1"]):
    import seaborn as sns
    import matplotlib.pyplot as plt
    df_filtered = _filter_duplicates_by_id(df, keep_id=plot_id)

    # ensure model column sorted nicely for consistent colors
    df_filtered["model"] = df_filtered["model"].astype("category")
    model_order = sorted(df_filtered["model"].unique())
    palette = sns.color_palette("Set2", len(model_order))

    # melt for plotting
    melted = df_filtered.melt(
        id_vars=["model", "data"],
        value_vars=metrics,
        var_name="metric",
        value_name="score"
    )

    # helper to make consistent chart layout
    def plot_metrics(subset, title, emphasize=None):
        plt.figure(figsize=(12,6))
        sns.barplot(
            data=subset,
            x="metric", y="score",
            hue="model",
            palette=palette,
            errorbar=None
        )
        plt.ylim(0, 1)
        plt.title(title, fontsize=14, weight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        if emphasize:
            plt.axhline(0, color="none") 
            # slightly bold recall bars
            bars = plt.gca().patches
            for b in bars:
                if b.get_x() < 0:  #type: ignore
                    continue
            # recolor recall tick label
            xticks = plt.gca().get_xticklabels()
            for t in xticks:
                if t.get_text().lower() == emphasize.lower():
                    t.set_weight("bold") #type: ignore
                    t.set_color("#b22222")
        plt.legend(title="Model", frameon=False)
        plt.tight_layout()
        plt.show()


    # --- Plot 1: Validation (pre-threshold) ---
    validate_df = melted[melted["data"] == "validate"]
    plot_metrics(validate_df,
                "Model Comparison on Validation Set (Pre-Threshold Testing)")

    # --- Plot 2: Test (F2-weighted, recall-focused) ---
    test_df = melted[melted["data"] == "test"]
    plot_metrics(test_df,
                "Model Comparison on Held-Out Test Set (F2-Weighted Threshold)",
                emphasize="recall")
    

def summary_by_auc(df, plot_id):
    from IPython.display import display

    df_collapse = _filter_duplicates_by_id(df, plot_id)

    df_collapse = df_collapse.sort_values(by='roc_auc', ascending=False)

    df_collapse = df_collapse = df_collapse[df_collapse['data'] == 'test']

    df_collapse['roc_auc'] = np.round(df_collapse['roc_auc'], 3)

    print('Models by ROC_AUC')
    display(df_collapse.head(5)[['model', 'roc_auc']])

def summary_by_recall(df):
    from IPython.display import display
    df_collapse = df.sort_values(by='recall', ascending=False)
    df_collapse = df_collapse = df_collapse[df_collapse['data'] == 'test']
    df_collapse['recall'] = np.round(df_collapse['recall'], 3)

    print('Models by Recall')
    display(df_collapse.head(5)[['model', 'recall']])