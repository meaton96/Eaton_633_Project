import matplotlib.pyplot as plt
import seaborn as sns

def plot_pr_with_break_even(curves, R=16300.0, e=0.50, c_m_list=(500.0, 1000.0, 1500.0), title=None):
    # PR: x=recall, y=precision
    plt.figure()
    plt.plot(curves["recall"], curves["precision"], label="PR curve")
    # horizontal break-even precision lines: PPV_break_even = c_m / (e * R)
    for c in c_m_list:
        y_be = c / (e * R)
        plt.hlines(y_be, xmin=0, xmax=1, linestyles="--", label=f"Break-even PPV @ ${int(c)}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    if title: plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_savings_vs_threshold(curves, c_m_list=(500.0, 1000.0, 1500.0), per_1000=True, title=None):
    scale = 1000.0 if per_1000 else 1.0
    plt.figure()
    for c in c_m_list:
        plt.plot(curves["thresholds"], scale * curves["savings"][c], label=f"${int(c)}")
    plt.xlabel("Threshold")
    plt.ylabel(("Savings per 1000 patients ($)" if per_1000 else "Savings per patient ($)"))
    if title: plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_y_dist(y, ax=None, title='Target Variable Distribution'):
    ax = ax or plt.gca()
    sns.countplot(x=y, hue=y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Readmitted')
    ax.set_ylabel('count')

def check_y_dist(y_train, y_validate, y_test):
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    plot_y_dist(y_train, ax=axes[0], title='Train')
    plot_y_dist(y_validate, ax=axes[1], title='Validate')
    plot_y_dist(y_test, ax=axes[2], title='Test')

    plt.tight_layout()
    plt.show()
    

def plot_metrics(df, plot_id, title_suff = "", metrics = ["roc_auc", "accuracy", "precision", "recall", "f1"]):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from shared_util.metrics.util import filter_duplicates_by_id
    df_filtered = filter_duplicates_by_id(df, keep_id=plot_id)

    # ensure model column sorted nicely for consistent colors
    df_filtered["model"] = df_filtered["model"].astype("category")
    model_order = sorted(df_filtered["model"].unique())
    palette = sns.color_palette("Set1", len(model_order))

    # melt for plotting
    melted = df_filtered.melt(
        id_vars=["model", "data"],
        value_vars=metrics,
        var_name="metric",
        value_name="score"
    )

    # --- Plot 1: Validation (pre-threshold) ---
    validate_df = melted[melted["data"] == "validate"]
    _plot_metrics(validate_df,
                f"Model Comparison on Validation Set (Pre-Threshold Testing) [{title_suff}]", palette)

    # --- Plot 2: Test
    test_df = melted[melted["data"] == "test"]
    _plot_metrics(test_df,
                f"Model Comparison on Held-Out Test Set (Cost Weighted Threshold) [{title_suff}]", palette)
    
# helper to make consistent chart layout
def _plot_metrics(subset, title, palette):
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
    plt.legend(title="Model", frameon=False)
    plt.tight_layout()
    plt.show()