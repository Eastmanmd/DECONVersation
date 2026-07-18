import os
import numpy as np
import pandas as pd
from scipy.optimize import nnls, minimize
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
            #if normalize and coeffs.sum() > 0:
            #    coeffs = coeffs / coeffs.sum()
        
        elif solver == "nnls_mod":
            X_aug = np.vstack([X, 1000 * np.ones((1, X.shape[1]))])
            y_aug = np.append(y, 1000)
            coeffs, _ = nnls(X_aug, y_aug)
        
        elif solver == "dwls":
            coeffs, _ = nnls(X, y)  # initial fit
            for _ in range(4):
                y_hat = X @ coeffs
                y_hat = np.clip(y_hat, 1e-6, None)
                w = 1.0 / (y_hat ** 2)
                lo, hi = np.quantile(w, [0.05, 1 - 0.05])
                w = np.clip(w, lo, hi)
                sw = np.sqrt(w)
                coeffs, _ = nnls(X * sw[:, None], y * sw)
        
        elif solver == "simplex":
            coeffs = _simplex_ls(X, y)

        elif solver == "ridge_simplex":
            coeffs = _simplex_ls(X, y, alpha=1.0)

        elif solver == "dwls_simplex":
            coeffs = _simplex_ls(X, y)  # initial fit
            for _ in range(4):
                y_hat = np.clip(X @ coeffs, 1e-6, None)
                w = 1.0 / (y_hat ** 2)
                lo, hi = np.quantile(w, [0.05, 1 - 0.05])
                w = np.clip(w, lo, hi)
                coeffs = _simplex_ls(X, y, weights=w)
        
        elif solver == "ridge":
            model = Ridge(alpha=1.0, positive=True, fit_intercept=False)
            coeffs = model.fit(X, y).coef_

        elif solver == "elasticnet":
            model = ElasticNet(alpha=0.1, l1_ratio=0.5, positive=True, fit_intercept=False)
            coeffs = model.fit(X, y).coef_
        
        elif solver == "nusvr":
            model = NuSVR(kernel='linear', nu=0.5, C=1.0)
            coeffs = model.fit(X, y).coef_.ravel()
            coeffs = np.clip(coeffs, 0, None)
        
        if normalize and coeffs.sum() > 0:
            coeffs = coeffs / coeffs.sum()
        
        proportions.append(coeffs)

    proportions_df = pd.DataFrame(
        proportions,
        index=bulk_df.columns,
        columns=celltypes,
    )

    return proportions_df

def _simplex_ls(X, y, weights=None, alpha=0.0, prior=None):
        n = X.shape[1]
        p0 = np.full(n, 1.0 / n)
        w = weights if weights is not None else np.ones_like(y)

        def obj(p):
            resid = X @ p - y
            wr = w * resid
            val = wr @ resid
            grad = 2 * X.T @ wr
            if alpha > 0:
                pr = prior if prior is not None else np.full(n, 1.0 / n)
                val += alpha * np.sum((p - pr) ** 2)
                grad += 2 * alpha * (p - pr)
            return val, grad

        cons = [{"type": "eq", "fun": lambda p: p.sum() - 1.0,
                 "jac": lambda p: np.ones(n)}]
        res = minimize(obj, p0, jac=True, bounds=[(0, None)] * n,
                        constraints=cons, method="SLSQP",
                        options={"maxiter": 500, "ftol": 1e-10})
        return res.x
