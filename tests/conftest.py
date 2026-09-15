from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse


@pytest.fixture
def signature_df() -> pd.DataFrame:
    """Three features by two cell types with an identifiable mixture."""
    return pd.DataFrame(
        {
            "type_a": [1.0, 0.0, 1.0],
            "type_b": [0.0, 1.0, 1.0],
        },
        index=["gene_1", "gene_2", "gene_3"],
    )


@pytest.fixture
def bulk_df() -> pd.DataFrame:
    """Known 70% type_a / 30% type_b mixture."""
    return pd.DataFrame(
        {"sample_1": [0.7, 0.3, 1.0]},
        index=["gene_1", "gene_2", "gene_3"],
    )


def _small_adata(*, as_sparse: bool) -> ad.AnnData:
    values = np.array(
        [
            [2, 0, 1],
            [4, 0, 3],
            [0, 6, 1],
            [0, 2, 3],
        ],
        dtype=float,
    )
    matrix = sparse.csr_matrix(values) if as_sparse else values
    obs = pd.DataFrame(
        {
            "cell_type": ["type_a", "type_a", "type_b", "type_b"],
            "sample": ["donor_1", "donor_2", "donor_1", "donor_2"],
        },
        index=["cell_1", "cell_2", "cell_3", "cell_4"],
    )
    var = pd.DataFrame(index=["gene_1", "gene_2", "gene_3"])
    return ad.AnnData(X=matrix, obs=obs, var=var)


@pytest.fixture
def dense_adata() -> ad.AnnData:
    return _small_adata(as_sparse=False)


@pytest.fixture
def sparse_adata() -> ad.AnnData:
    return _small_adata(as_sparse=True)
