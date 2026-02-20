import pickle
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import scipy.sparse as sp 
import matplotlib.pyplot as plt
import anndata

def load_and_prep_data(
    path: str,
    mode: str,
    organism: str,
    cell_type_col: str,
):
    """
    Load and prepare an AnnData object for Geneformer or Cell2Sentence workflows.

    Parameters
    ----------
    path : str
        Path to input .h5ad file.
    mode : str
        Preparation mode. Must be either:
        - "geneformer"
        - "c2s"
    organism : str
        Organism name (required for Cell2Sentence).
        Example: "Homo sapiens"
    cell_type_col : str
        Column name in adata.obs containing cell type annotations.

    Returns
    -------
    adata : AnnData
        Processed AnnData object formatted for the selected embedding model.

    """

    import scanpy as sc
    import numpy as np

    # -----------------------------
    # Load AnnData object
    # -----------------------------
    adata = sc.read_h5ad(path)

    # Validate required column exists
    if cell_type_col not in adata.obs.columns:
        raise ValueError(
            f"'{cell_type_col}' not found in adata.obs columns."
        )

    # Geneformer Preparation
    if mode == "geneformer":

        adata.obs["cell_id"] = adata.obs.index
        adata.obs["cell_type"] = adata.obs[cell_type_col]

        # Get var.index
        var_index = adata.var.index.astype(str)

        if not all(
            gene.startswith("ENSG") or gene.startswith("ENSMUSG")
            for gene in var_index
        ):
            raise ValueError(
                "adata.var.index must contain Ensembl gene IDs "
                "(ENSG for human or ENSMUSG for mouse)."
            )

        # Strip version numbers
        adata.var["ensembl_id"] = var_index.str.split(".").str[0]

        # Compute total counts per cell
        adata.obs["n_counts"] = np.array(
            adata.X.sum(axis=1)
        ).flatten()


    # Cell2Sentence Preparation
    elif mode == "c2s":

        # Required metadata fields for C2S arrow conversion
        adata.obs["organism"] = organism
        adata.obs["cell_type"] = adata.obs[cell_type_col]
        adata.obs["tissue"] = "unknown"
        adata.obs["sex"] = "unknown"
        adata.obs["batch_condition"] = "unknown"

    else:
        raise ValueError(
            "mode must be either 'geneformer' or 'c2s'"
        )

    return adata


def gene_id_name_map()



    