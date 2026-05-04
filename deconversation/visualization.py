# ============================================
# Required Imports
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


# ============================================
# Plot True vs Predicted
# ============================================
def plot_true_vs_predicted(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    n_cols: int = 3,
    figsize_per_plot: tuple = (3, 3),
    dot_size: int = 12,
    stratify_by_celltype: bool = True,
    method: str = "pearson",
    save_path: str | None = None,
):
    """
    Plot true vs predicted proportions.

    Parameters
    ----------
    y_true_df : pd.DataFrame 
        Ground truth proportions (samples × cell types).

    y_pred_df : pd.DataFrame
        Predicted proportions (samples × cell types).

    n_cols : int, default=6
        Number of columns in subplot grid 
        Only used if stratify_by_celltype=True

    figsize_per_plot : tuple, default=(3,3)
        Size (width, height) per subplot.

    dot_size : int, default=12
        Size of scatter points.

    stratify_by_celltype : bool, default=True
        If True: one subplot per cell type.
        If False: single aggregated plot (all values combined).

    method : str, default="pearson"
        Correlation method:
            - "pearson"
            - "spearman"

    save_path : str, optional
        If provided, saves figure to this path.

    Returns
    -------
    Scatter plot
    """

    # Ensure input method is valid
    if method not in ["pearson", "spearman"]:
        raise ValueError("method must be 'pearson' or 'spearman'.")

    # Align samples and cell types
    common_samples = y_true_df.index.intersection(y_pred_df.index)
    common_cols = y_true_df.columns.intersection(y_pred_df.columns)

    y_true_df = y_true_df.loc[common_samples, common_cols]
    y_pred_df = y_pred_df.loc[common_samples, common_cols]

    # --------------------------------------------------
    # Single aggregated plot
    # --------------------------------------------------
    if not stratify_by_celltype:

        # add cell type column    
        df_true = y_true_df.reset_index().melt(id_vars=y_true_df.index.name or "index",
                                               var_name="cell_type",
                                               value_name="true")
        
        df_pred = y_pred_df.reset_index().melt(id_vars=y_pred_df.index.name or "index",
                                               var_name="cell_type",
                                               value_name="pred")
        # Merge true + predicted
        df_plot = df_true.merge(df_pred, on=[y_true_df.index.name or "index", "cell_type"])

        #x = y_true_df.values.flatten()
        #y = y_pred_df.values.flatten()

        plt.figure(figsize=(figsize_per_plot[0]*1.5,
                            figsize_per_plot[1]*1.5))

        sns.scatterplot(data=df_plot,
                        x="true",
                        y="pred",
                        hue="cell_type",
                        alpha=0.7,
                        s=dot_size)

        # Identity line
        lo, hi = min(df_plot["true"].min(), df_plot["pred"].min()), max(df_plot["true"].max(), df_plot["pred"].max())        
        plt.plot([lo, hi], [lo, hi], "r--")

        # Correlation
        if method == "pearson":
            corr, _ = pearsonr(df_plot["true"], df_plot["pred"])
        else:
            corr, _ = spearmanr(df_plot["true"], df_plot["pred"])

        plt.text(
            0.05, 0.95,
            f"{method.capitalize()} r = {corr:.2f}",
            transform=plt.gca().transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
        )

        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(" ")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")

        plt.show()
        return

    # --------------------------------------------------
    # Stratified by cell type
    # --------------------------------------------------
    cols = common_cols #( columns are cell types)
    n_rows = int(np.ceil(len(cols) / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_per_plot[0],
                 n_rows * figsize_per_plot[1]),
    )

    axes = np.array(axes).ravel()

    for ax, col in zip(axes, cols):

        x = y_true_df[col]
        y = y_pred_df[col]

        sns.scatterplot(x=x, y=y, alpha=0.7, s=dot_size, ax=ax)

        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "r--")

        if method == "pearson":
            corr, _ = pearsonr(x, y)
        else:
            corr, _ = spearmanr(x, y)

        ax.text(
            0.05, 0.95,
            f"r = {corr:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
        )

        ax.set(title=col, xlabel="True", ylabel="Predicted")

    # Remove unused axes
    for ax in axes[len(cols):]:
        fig.delaxes(ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    plt.show()
    