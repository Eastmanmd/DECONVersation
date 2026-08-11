import os
import numpy as np
import pandas as pd
from scipy.optimize import nnls, minimize
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import NuSVR
from typing import Dict, List, Optional
from time import perf_counter
import torch
import torch.nn.functional as F

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
    
    if solver == "gradient_descent":
        y = bulk_df.values
        coeffs = gd_decompose(X, y, loss="cosine", sum_to_one=True, regularization="l1", lam=0.01)
        proportions_df = pd.DataFrame(
                coeffs,
                index=bulk_df.columns,
                columns=celltypes)
        return proportions_df

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
            coeffs = solve_dampened_wls(X, y)

        elif solver == "dwls_approx":
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

        elif solver == "centered_simplex":
            coeffs = centered_simplex_deconvolution(X,y)
        #elif solver == "gradient_descent":
        #    coeffs = gd_decompose(X, y, loss="cosine", sum_to_one=True, regularization="l1", lam=0.01)

        if normalize and coeffs.sum() > 0:
            coeffs = coeffs / coeffs.sum()
        
        proportions.append(coeffs)

    proportions_df = pd.DataFrame(
        proportions,
        index=bulk_df.columns,
        columns=celltypes,
    )

    return proportions_df

def gd_decompose(
    references,
    mixture,
    n_iter=1000,
    lr=0.05,
    loss="cosine",        # "mse" or "cosine"
    regularization=None,  # None, "l1", or "entropy"
    lam=0.01,
    sum_to_one=True,
    tol=1e-6, patience=10
):  
    references = F.normalize(torch.tensor(references.T, dtype=torch.float32), dim=-1)
    references = torch.tensor(references, dtype=torch.float32)
    mixture = F.normalize(torch.tensor(mixture.T, dtype=torch.float32), dim=-1)
    mixture = torch.tensor(mixture, dtype=torch.float32)

    batched = mixture.ndim == 2
    if not batched:
        mixture = mixture.unsqueeze(0)

    n_batch = mixture.shape[0]
    n_sources = references.shape[0]

    # Initialize weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    references = references.to(device)
    mixture = mixture.to(device)
    w = torch.zeros(n_batch, n_sources, requires_grad=True, device=device)
    #w = torch.zeros(n_batch, n_sources, requires_grad=True)
    #optimizer = torch.optim.Adam([w], lr=lr)
    optimizer = torch.optim.Adam([w], lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter) 
    prev_loss = float("inf")
    for i in range(n_iter):
        optimizer.zero_grad()
        w_pos = F.softplus(w)

        if sum_to_one:
            w_pos = w_pos / w_pos.sum(dim=-1, keepdim=True)

        recon = w_pos @ references  # (batch, n_sources) x (n_sources, embed_dim)

        if loss == "cosine":
            loss_val = 1 - F.cosine_similarity(recon, mixture, dim=-1).mean()
        else:
            loss_val = F.mse_loss(recon, mixture)

        if regularization == "l1":
            loss_val = loss_val + lam * w_pos.mean()
        elif regularization == "entropy":
            p = w_pos / (w_pos.sum(dim=-1, keepdim=True) + 1e-8)
            loss_val = loss_val - lam * (-(p * (p + 1e-8).log()).sum(dim=-1).mean())

        loss_val.backward()
        optimizer.step()
        scheduler.step()

        if i % patience == 0:
            if abs(prev_loss - loss_val.item()) < 1e-6:
                break
            prev_loss = loss_val.item()

    with torch.no_grad():
        w_final = F.softplus(w)
        if sum_to_one:
            w_final = w_final / w_final.sum(dim=-1, keepdim=True)

    return w_final.squeeze(0).numpy() if not batched else w_final.numpy()

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
            "nnls", "nnls_mod", "dwls", "dwls_approx",
            "simplex", "ridge_simplex", "dwls_simplex",
            "ridge", "elasticnet", "nusvr", "simplex_nnls",
            "gradient_descent","centered_simplex"
        ]

    # checks
    check_signature(signature_df.T)
        
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
    """
    Inspect signatures with rows=cell types and columns=embedding dimensions.
    """

    signature_df = signature_df.loc[
        :, signature_df.ne(0).any(axis=0)
    ].copy()

    S = signature_df.to_numpy(dtype=float)

    row_norms = np.linalg.norm(S, axis=1)
    zero_cell_types = signature_df.index[np.isclose(row_norms, 0)]

    if len(zero_cell_types):
        raise ValueError(
            "Zero-norm cell-type signatures: "
            f"{zero_cell_types.tolist()}"
        )

    S_unit = S / row_norms[:, None]

    singular_values = np.linalg.svd(S_unit, compute_uv=False)
    condition_number = np.linalg.cond(S_unit)
    smallest_singular_value = singular_values[-1]

    cosine_matrix = S_unit @ S_unit.T
    np.fill_diagonal(cosine_matrix, -np.inf)

    i, j = np.unravel_index(cosine_matrix.argmax(), cosine_matrix.shape)
    max_pairwise_cosine_similarity = cosine_matrix[i, j]

    print(
        f"Condition number: {condition_number:.2f}.\n"
        "~1-10: signatures are well separated; NNLS is usually hard "
        "to improve materially with another solver."
    )
    print(
        f"Smallest singular value: {smallest_singular_value:.4g}.\n"
        "If value is close to zero, the signature matrix is "
        "ill-conditioned; estimated proportions may be unstable."
    )
    print(
        f"Max pairwise cosine similarity between cell types: "
        f"{max_pairwise_cosine_similarity:.4f}.\n"
        f"Most similar pair: {signature_df.index[i]} vs "
        f"{signature_df.index[j]}.\n"
        "If value is close to 1, these cell types are highly "
        "similar and difficult to resolve separately."
    )

    return {
        "condition_number": condition_number,
        "smallest_singular_value": smallest_singular_value,
        "max_pairwise_cosine_similarity": max_pairwise_cosine_similarity,
        "most_similar_pair": (
            signature_df.index[i],
            signature_df.index[j],
        ),
    }

def centered_simplex_deconvolution(
    signature: np.ndarray,  # embedding dimensions x cell types
    bulk: np.ndarray,       # embedding dimensions
) -> np.ndarray:
    n_cell_types = signature.shape[1]

    centroid = signature.mean(axis=1)
    signature_centered = signature - centroid[:, None]
    bulk_centered = bulk - centroid

    def objective(p: np.ndarray) -> float:
        residual = bulk_centered - signature_centered @ p
        return float(residual @ residual)

    result = minimize(
        objective,
        x0=np.full(n_cell_types, 1.0 / n_cell_types),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_cell_types,
        constraints={
            "type": "eq",
            "fun": lambda p: p.sum() - 1.0,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Simplex optimization failed: {result.message}"
        )

    return result.x

def solve_ols_internal(S: np.ndarray, B: np.ndarray) -> np.ndarray:
    solution, _ = nnls(S, B)
    return solution


def solve_dampened_wls_j(S: np.ndarray, B: np.ndarray,
                          gold_standard: np.ndarray, j: int) -> np.ndarray:
    multiplier = 1 * (2 ** (j - 1))
    sol = gold_standard
    pred = S @ sol
    pred = np.where(np.abs(pred) < 1e-6, np.sign(pred) * 1e-6 + 1e-6, pred)
    ws = (1.0 / pred) ** 2          # weights from current solution's prediction
    ws_scaled = ws / ws.min()
    ws_dampened = ws_scaled.copy()
    ws_dampened[ws_scaled > multiplier] = multiplier   # cap only, no floor

    sw = np.sqrt(ws_dampened)
    # weighted NNLS via sqrt(w) reweighting — equivalent to the R solve.QP
    # on D = S'WS, d = S'WB with w >= 0 constraint
    solution, _ = nnls(S * sw[:, None], B * sw)
    return solution


def find_dampening_constant(S: np.ndarray, B: np.ndarray,
                             gold_standard: np.ndarray,
                             n_boot: int = 100) -> int:
    sol = gold_standard
    pred = S @ sol
    pred = np.where(np.abs(pred) < 1e-6, np.sign(pred) * 1e-6 + 1e-6, pred)
    ws = (1.0 / (pred)) ** 2
    ws_scaled = ws / ws.min()

    ws_scaled_finite = ws_scaled[np.isfinite(ws_scaled)]
    if len(ws_scaled_finite) == 0 or ws_scaled_finite.max() <= 1:
        return 1

    n_genes = len(ws)
    max_j = int(np.ceil(np.log2(ws_scaled_finite.max())))

    mean_variance_per_j = []

    for j in range(1, max_j + 1):
        multiplier = 1 * (2 ** (j - 1))
        ws_dampened = ws_scaled.copy()
        ws_dampened[ws_scaled > multiplier] = multiplier

        boot_solutions = []
        rng_master = np.random.RandomState()  # reseeded per iter below, matches R's set.seed(i)
        for seed in range(1, n_boot + 1):
            rng = np.random.RandomState(seed)
            subset = rng.choice(n_genes, size=int(n_genes * 0.5), replace=False)

            # R: fit <- lm(B[subset] ~ -1 + S[subset,], weights = wsDampened[subset])
            # unconstrained weighted least squares (intentionally NOT non-negative)
            w_sub = ws_dampened[subset]
            sw_sub = np.sqrt(w_sub)
            Xw = S[subset] * sw_sub[:, None]
            yw = B[subset] * sw_sub
            coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

            coef_sum = coef.sum()
            if coef_sum == 0:
                continue  # avoid div by zero; R would produce NaN/Inf here too
            scaled_sol = coef * gold_standard.sum() / coef_sum
            boot_solutions.append(scaled_sol)

        if len(boot_solutions) == 0:
            mean_variance_per_j.append(np.inf)
            continue

        boot_solutions = np.array(boot_solutions)         # (n_boot, n_celltypes)
        per_gene_sd = boot_solutions.std(axis=0, ddof=1)   # sd across bootstraps, per cell type
        mean_variance_per_j.append(np.mean(per_gene_sd ** 2))

    best_j = int(np.argmin(mean_variance_per_j)) + 1  # +1 since j starts at 1
    return best_j


def solve_dampened_wls(S: np.ndarray, B: np.ndarray,
                        max_iter: int = 1000, tol: float = 0.01) -> np.ndarray:
    solution = solve_ols_internal(S, B)

    j = find_dampening_constant(S, B, solution)

    change = 1.0
    iterations = 0
    while change > tol and iterations < max_iter:
        new_solution = solve_dampened_wls_j(S, B, solution, j)

        # R: solutionAverage <- rowMeans(cbind(newsolution, matrix(solution, ncol=4)))
        # = (newsolution + 4*solution) / 5  -- heavy damping toward old solution
        solution_avg = (new_solution + 4 * solution) / 5

        change = np.linalg.norm(solution_avg - solution)
        solution = solution_avg
        iterations += 1

    return solution / solution.sum()   # normalized to proportions, matches R output
