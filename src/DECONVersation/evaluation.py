import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ============================================
# Compute Root Mean Squared Error
# ============================================
def compute_rmse(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    return_per_sample: bool = False,
    return_per_celltype: bool = False,
):
    """
    Compute RMSE between true and predicted cell-type proportions.

    Parameters
    ----------
    true_df : pd.DataFrame
        Ground truth proportions.
        Shape: (samples × cell types)

    pred_df : pd.DataFrame
        Predicted proportions.
        Shape: (samples × cell types)

    return_per_sample : bool, default=False
        If True, also return RMSE per sample.

    return_per_celltype : bool, default=False
        If True, also return RMSE per cell type.

    Returns
    -------
    float or dict
        If no optional flags:
            Overall RMSE 

        If flags enabled:
            Dictionary containing:
                - "overall"
                - "per_sample" (optional)
                - "per_celltype" (optional)
    """

    # Ensure input is a dataframe
    if not isinstance(true_df, pd.DataFrame) or not isinstance(pred_df, pd.DataFrame):
        raise TypeError("Inputs must be pandas DataFrames.")


    # Align samples and cell types
    common_samples = true_df.index.intersection(pred_df.index)
    common_celltypes = true_df.columns.intersection(pred_df.columns)

    if len(common_samples) == 0 or len(common_celltypes) == 0:
        raise ValueError("No overlapping samples or cell types found.")

    true_aligned = true_df.loc[common_samples, common_celltypes]
    pred_aligned = pred_df.loc[common_samples, common_celltypes]

    # Compute errors
    diff = true_aligned.values - pred_aligned.values
    mse = np.mean(diff ** 2)
    overall_rmse = np.sqrt(mse)

    results = {"overall": overall_rmse}

    # Per-sample RMSE
    if return_per_sample:
        sample_rmse = np.sqrt(np.mean(diff ** 2, axis=1))
        results["per_sample"] = pd.Series(
            sample_rmse,
            index=common_samples,
            name="rmse",
        )

    # Per-celltype RMSE
    if return_per_celltype:
        celltype_rmse = np.sqrt(np.mean(diff ** 2, axis=0))
        results["per_celltype"] = pd.Series(
            celltype_rmse,
            index=common_celltypes,
            name="rmse",
        )

    if return_per_sample or return_per_celltype:
        return results
    else:
        return overall_rmse



# ============================================
# Compute Correlation
# ============================================
def compute_correlation(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    method: str = "pearson",
    return_per_sample: bool = False,
    return_per_celltype: bool = False,
):
    """
    Compute correlation between true and predicted cell-type proportions.

    Parameters
    ----------
    true_df : pd.DataFrame
        Ground truth proportions.
        Shape: (samples × cell types)

    pred_df : pd.DataFrame
        Predicted proportions.
        Shape: (samples × cell types)

    method : str, default="pearson"
        Correlation method:
            - "pearson"
            - "spearman"

    return_per_sample : bool, default=False
        If True, return correlation per sample.

    return_per_celltype : bool, default=False
        If True, return correlation per cell type.

    Returns
    -------
    float or dict
        If no optional flags:
            Overall correlation

        If flags enabled:
            Dictionary containing:
                - "overall"
                - "per_sample" (optional)
                - "per_celltype" (optional)

    """

    # Ensure input is a dataframe
    if not isinstance(true_df, pd.DataFrame) or not isinstance(pred_df, pd.DataFrame):
        raise TypeError("Inputs must be pandas DataFrames.")
        
    # Ensure input method is valid
    if method not in ["pearson", "spearman"]:
        raise ValueError("method must be either 'pearson' or 'spearman'.")

    
    # Align samples and cell types
    common_samples = true_df.index.intersection(pred_df.index)
    common_celltypes = true_df.columns.intersection(pred_df.columns)

    if len(common_samples) == 0 or len(common_celltypes) == 0:
        raise ValueError("No overlapping samples or cell types found.")

    true_aligned = true_df.loc[common_samples, common_celltypes]
    pred_aligned = pred_df.loc[common_samples, common_celltypes]

    
    # Overall correlation
    true_flat = true_aligned.values.flatten()
    pred_flat = pred_aligned.values.flatten()

    if method == "pearson":
        overall_corr = np.corrcoef(true_flat, pred_flat)[0, 1]
    else:
        overall_corr = pd.Series(true_flat).corr(
            pd.Series(pred_flat),
            method="spearman"
        )

    results = {"overall": overall_corr}

    # --------------------------------------------------
    # Per-sample correlation
    # --------------------------------------------------
    if return_per_sample:
        sample_corr = []
        for i in range(len(common_samples)):
            if method == "pearson":
                r = np.corrcoef(
                    true_aligned.iloc[i],
                    pred_aligned.iloc[i]
                )[0, 1]
            else:
                r = true_aligned.iloc[i].corr(
                    pred_aligned.iloc[i],
                    method="spearman"
                )
            sample_corr.append(r)

        results["per_sample"] = pd.Series(
            sample_corr,
            index=common_samples,
            name="correlation",
        )

    # --------------------------------------------------
    # Per-celltype correlation
    # --------------------------------------------------
    if return_per_celltype:
        celltype_corr = []
        for col in common_celltypes:
            if method == "pearson":
                r = np.corrcoef(
                    true_aligned[col],
                    pred_aligned[col]
                )[0, 1]
            else:
                r = true_aligned[col].corr(
                    pred_aligned[col],
                    method="spearman"
                )
            celltype_corr.append(r)

        results["per_celltype"] = pd.Series(
            celltype_corr,
            index=common_celltypes,
            name="correlation",
        )

    if return_per_sample or return_per_celltype:
        return results
    else:
        return overall_corr



# ============================================
# Plot RMSE vs Correlation Per Cell Type
# ============================================
def plot_rmse_vs_corr_by_celltype(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    method: str = "pearson",
    dot_size: int = 40,
    annotate: bool = True,
    save_path: str | None = None,
):
    """
    Compute RMSE and correlation per cell type (class),
    then plot correlation (x-axis) vs RMSE (y-axis).

    Parameters
    ----------
    true_df : pd.DataFrame
        Ground truth proportions (samples × cell types)

    pred_df : pd.DataFrame
        Predicted proportions (samples × cell types)

    method : str, default="pearson"
        Correlation method:
            - "pearson"
            - "spearman"

    dot_size : int, default=60
        Size of scatter points.

    annotate : bool, default=True
        Whether to label points with class names.

    save_path : str, optional
        If provided, saves figure to this path.

    Returns
    -------
    pd.DataFrame
        DataFrame containing RMSE and correlation per class.
    """

    # Ensure corrected method is selected
    if method not in ["pearson", "spearman"]:
        raise ValueError("method must be 'pearson' or 'spearman'.")

    # Align samples and cell types
    common_samples = true_df.index.intersection(pred_df.index)
    common_cols = true_df.columns.intersection(pred_df.columns)

    if len(common_samples) == 0 or len(common_cols) == 0:
        raise ValueError("No overlapping samples or cell types found.")

    true_aligned = true_df.loc[common_samples, common_cols]
    pred_aligned = pred_df.loc[common_samples, common_cols]

    results = []

    for col in common_cols:
        x = true_aligned[col].values
        y = pred_aligned[col].values

        # RMSE
        rmse = np.sqrt(np.mean((x - y) ** 2))

        # Correlation
        if method == "pearson":
            corr, _ = pearsonr(x, y)
        else:
            corr, _ = spearmanr(x, y)

        results.append((col, rmse, corr))

    metrics_df = pd.DataFrame(
        results,
        columns=["cell_type", "rmse", "correlation"]
    ).set_index("cell_type")

    # Plot
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=metrics_df,
        x="correlation",
        y="rmse",
        s=dot_size
    )

    if annotate:
        for cell_type, row in metrics_df.iterrows():
            plt.text(
                row["correlation"],
                row["rmse"],
                cell_type,
                fontsize=14,
                ha="left",
                va="bottom"
            )

    plt.xlabel(f"Cor ({method})", fontsize = 18)
    plt.ylabel("RMSE", fontsize = 18)
    plt.title(" ")

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    plt.show()

