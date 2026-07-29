import os
import numpy as np
import pandas as pd
from scipy.optimize import nnls, minimize
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import NuSVR
from typing import Dict, List, Optional
from time import perf_counter

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

    #print(f"Using {len(common_features)} common features.")

    # --------------------------------------------------
    # Run NNLS 
    # --------------------------------------------------
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
        
        elif solver == "simplex_nnls":
            coeffs = simplex_nnls(X, y)

        if normalize and coeffs.sum() > 0:
            coeffs = coeffs / coeffs.sum()
        
        proportions.append(coeffs)

    proportions_df = pd.DataFrame(
        proportions,
        index=bulk_df.columns,
        columns=celltypes,
    )

    return proportions_df

def simplex_nnls(X, y, maxiter=500, ftol=1e-10):
    """Fit nonnegative proportions that sum to one."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be (features, cell_types) and match y.")

    n_cell_types = X.shape[1]
    hessian = X.T @ X
    linear_term = X.T @ y

    def objective(p):
        return 0.5 * p @ hessian @ p - linear_term @ p

    def gradient(p):
        return hessian @ p - linear_term

    result = minimize(
        objective,
        x0=np.full(n_cell_types, 1.0 / n_cell_types),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, None)] * n_cell_types,
        constraints={
            "type": "eq",
            "fun": lambda p: p.sum() - 1.0,
            "jac": lambda p: np.ones_like(p),
        },
        options={"maxiter": maxiter, "ftol": ftol},
    )

    if not result.success:
        raise RuntimeError(f"Simplex NNLS failed: {result.message}")

    proportions = np.clip(result.x, 0.0, None)
    return proportions / proportions.sum()

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

def run_all_deconv(
    bulk_df: pd.DataFrame,
    signature_df: pd.DataFrame,
    solvers: Optional[List[str]] = None,
    normalize: bool = True,
    skip_errors: bool = True,
) -> Dict[str, pd.DataFrame]:

    # If none return all solvers 
    if solvers is None:
        solvers = [
            "nnls", "nnls_mod", "dwls",
            "simplex", "ridge_simplex", "dwls_simplex",
            "ridge", "elasticnet", "nusvr", "simplex_nnls"
        ]

    # checks
    check_signature(signature_df)
        
    results = {}  
    total_start = perf_counter()
    for solver in solvers:
        step_start = perf_counter()
        print(f"Running solver: {solver}")
        try:
            results[solver] = run_deconv(
                bulk_df, signature_df, solver=solver, normalize=normalize
            )
        except Exception as e:
            print("  [skipped] {} failed: {}: {}".format(
                solver, type(e).__name__, e))
            if not skip_errors:
                raise
        elapsed = perf_counter() - step_start
        print(
            f"Finished in {elapsed:.2f} seconds.",
            flush=True,
        )
    return results

def check_signature(signature_df: pd.DataFrame):
    X = signature_df.to_numpy(dtype=float)  # rows = features, columns = cell types

    column_norms = np.linalg.norm(X, axis=0)
    if np.any(column_norms == 0):
        raise ValueError("Signature matrix contains at least one zero-norm cell-type column.")

    X_unit = X / column_norms[None, :]

    singular_values = np.linalg.svd(X_unit, compute_uv=False)
    condition_number = singular_values[0] / singular_values[-1]
    smallest_singular_value = singular_values[-1]

    cosine_matrix = X_unit.T @ X_unit
    np.fill_diagonal(cosine_matrix, -np.inf)

    i, j = np.unravel_index(cosine_matrix.argmax(), cosine_matrix.shape)
    max_pairwise_cosine_similarity = cosine_matrix[i, j]

    print(f"Condition number: {condition_number:.2f}.\n~1–10: signatures are well separated; NNLS is usually hard to improve materially with another solver.")
    print(f"Smallest singular value: {smallest_singular_value:.4g}.\nIf the value is close to zero, the signature matrix is ill-conditioned; NNLS may be unstable and other solvers may be more robust.")
    print(
        f"Most similar pair: {signature_df.columns[i]} vs "
        f"{signature_df.columns[j]} "
        f"({max_pairwise_cosine_similarity:.4f})"
    )

    return {
        "condition_number": condition_number,
        "smallest_singular_value": smallest_singular_value,
        "max_pairwise_cosine_similarity": max_pairwise_cosine_similarity,
        "most_similar_pair": (
            signature_df.columns[i],
            signature_df.columns[j],
        ),
    }