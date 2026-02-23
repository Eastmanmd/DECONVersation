import os
import numpy as np
import pandas as pd
from scipy.optimize import nnls


# ============================================
# Non-Negative Least Squares (NNLS)
# ============================================
def run_deconv_nnls(
    bulk_df: pd.DataFrame,
    signature_df: pd.DataFrame,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Run NNLS-based deconvolution

    Parameters
    ----------
    bulk_df : pd.DataFrame
        bulk expression matrix (Rows:Genes, Columns:Samples)
        
    signature_df : pd.DataFrame
        Signature matrix (Rows:Genes, Columns:Cell types )

    normalize : bool, default=True
        If True, normalize NNLS coefficients to sum to 1 per sample
        so results represent proportions.

    Returns
    -------
    pd.DataFrame
        Estimated cell-type proportions (samples × cell types)

    Notes
    -----
    - Only features (genes/embeddings) shared between the two matrices
      are used.
    """

    #  Validate input bulk and signature matrices
    if not isinstance(bulk_df, pd.DataFrame):
        raise TypeError("bulk_df must be a pandas DataFrame.")

    if not isinstance(signature_df, pd.DataFrame):
        raise TypeError("signature_df must be a pandas DataFrame.")


    # Ensure genes/embeddings align
    common_features = bulk_df.index.intersection(signature_df.index)

    if len(common_features) == 0:
        raise ValueError(
            "No common genes (or embedding dimensions) found "
            "between bulk and signature matrix."
        )

    # Subset to common features
    bulk_df = bulk_df.loc[common_features]
    signature_df = signature_df.loc[common_features]

    print(f"Using {len(common_features)} common features.")

    # --------------------------------------------------
    # Run NNLS 
    # --------------------------------------------------
    print("Running NNLS deconvolution...")

    X = signature_df.values  
    celltypes = signature_df.columns

    proportions = []

    for sample in bulk_df.columns:
        y = bulk_df[sample].values  

        coeffs, _ = nnls(X, y)

        if normalize and coeffs.sum() > 0:
            coeffs = coeffs / coeffs.sum()

        proportions.append(coeffs)

    proportions_df = pd.DataFrame(
        proportions,
        index=bulk_df.columns,
        columns=celltypes,
    )

    return proportions_df

    