import os
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import NuSVR

# ============================================
# Non-Negative Least Squares (NNLS)
# ============================================
def run_deconv(
    bulk_df: pd.DataFrame,
    signature_df: pd.DataFrame,
    solver: str = "nnls",
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Run NNLS-based and other methods for deconvolution

    Parameters
    ----------
    bulk_df : pd.DataFrame
        bulk expression matrix (Rows:Genes, Columns:Samples)
        
    signature_df : pd.DataFrame
        Signature matrix (Rows:Genes, Columns:Cell types )
    
    solver : str
        solvers currently supported: nnls, ridge, elasticnet, nusvr

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
    print("Running deconvolution...")

    X = signature_df.values  
    celltypes = signature_df.columns

    proportions = []

    for sample in bulk_df.columns:
        y = bulk_df[sample].values  
        
        if solver == "nnls":
            coeffs, _ = nnls(X, y)
            if normalize and coeffs.sum() > 0:
                coeffs = coeffs / coeffs.sum()

        elif solver == "ridge":
            model = Ridge(alpha=1.0, positive=True)
            coeffs = model.fit(X, y).coef_

        elif solver == "elasticnet":
            model = ElasticNet(alpha=0.1, l1_ratio=0.5, positive=True)
            coeffs = model.fit(X, y).coef_
        
        elif solver == "nusvr":
            model = NuSVR(kernel='linear', nu=0.5, C=1.0)
            coeffs = model.fit(X, y).coef_.ravel()

        proportions.append(coeffs)

    proportions_df = pd.DataFrame(
        proportions,
        index=bulk_df.columns,
        columns=celltypes,
    )

    return proportions_df

    
