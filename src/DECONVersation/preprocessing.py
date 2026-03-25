import pickle
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import scipy.sparse as sp 
import matplotlib.pyplot as plt
import anndata

def load_and_prep_data(
    adata: ad.AnnData,
    mode: str,
    cell_type_col: str,
    organism: str = ""   
):
    """
    Load and prepare an AnnData object for Geneformer or Cell2Sentence workflows.

    Parameters
    ----------
    adata : str
        adata object
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

        # Require input organism 
        if organism.strip() == "":
            raise ValueError("`organism` must be provided, E.g 'Homo Sapiens' if human")

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

# --------------------------------
# Convert Ensembl to Symbol
# --------------------------------
def ensembl_to_symbol(gene_symbol_id_df, ensembl_list):
    mapping = dict(zip(gene_symbol_id_df["gene_id"], gene_symbol_id_df["gene_name"]))
    return [mapping.get(eid) for eid in ensembl_list]

    
# --------------------------------
# Convert symbol to ensembl
# --------------------------------
def symbol_to_ensembl(gene_symbol_id_df, symbol_list):
    mapping = dict(zip(gene_symbol_id_df["gene_name"], gene_symbol_id_df["gene_id"]))
    return [mapping.get(sym) for sym in symbol_list]


# ---------------------------------------------
# Mapping function to convert to symbol/ensembl
# ---------------------------------------------
def gene_id_name_map(
    gene_list,
    mode) :

    """
    Function to convert ensembl ids to symols or symbols to ids

    Parameters
    ----------
    gene_list : [str]
        Gene list (either ensembl ids or gene symbols)
        
    mode : str
        Preparation mode. Must be either:
        - "to_ensembl"
        - "to_symbol"

    Returns
    -------
    gene_list_mapped : [str]
        Gene list with mapped ensembl ids or symbols

    """

    # Load dataframe with both ensembl and gene_symbols
    gene_ids = pd.read_csv("/gpfs/commons/groups/compbio/projects/CZI_endom/RNA_temp/gene_names_gene_ids_czi_endo.csv", index_col=False)

    if mode == "to_ensembl":
        gene_list_mapped = symbol_to_ensembl(gene_ids, gene_list)

    elif mode == "to_symbol":
        gene_list_mapped = ensembl_to_symbol(gene_ids, gene_list)

    else:
        raise ValueError(
            "mode must be either 'to_ensembl' or 'to_symbol'"
        )

    return gene_list_mapped



# --------------------------------
# Create signature matrix 
# --------------------------------
def create_signature_matrix(
    adata,
    sample_col,
    celltype_col,
    groupby,
    sample_ids = None,
    celltypes = None,
    output_path=None,
):
    """
    Create a gene expression signature matrix from a single-cell AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        Input single-cell AnnData object.

    sample_col : str
        Column in `adata.obs` containing sample identifiers.

    sample_ids : list
        List of sample IDs to retain.

    celltype_col : str
        Column in `adata.obs` containing cell type labels.

    celltypes : list
        List of cell types to retain.

    groupby : str
        Column in `adata.obs` used to group cells when computing
        average gene expression (e.g., cell type).

    output_path : str, optional
        If provided, saves the resulting signature matrix as CSV.

    Returns
    -------
    pandas.DataFrame
        Signature matrix of shape (genes × groups),
        where rows are genes and columns are groups.
    """

    # ----------------------------
    # Prep data
    # ----------------------------

    # validate columns in anndata
    for col in [sample_col, celltype_col, groupby]:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs.")

    
    # Subset by  select samples
    if sample_ids is not None:
        adata = adata[adata.obs[sample_col].isin(sample_ids)].copy()
        if adata.n_obs == 0:
            raise ValueError("No cells remain after subsetting by sample_ids.")

    # Subset by select cell types
    if celltypes is not None:
        adata = adata[adata.obs[celltype_col].isin(celltypes)].copy()
        if adata.n_obs == 0:
            raise ValueError("No cells remain after subsetting by celltypes.")

    # Convert expression matrix
    if hasattr(adata.X, "toarray"):
        expr = pd.DataFrame(
            adata.X.toarray(),
            index=adata.obs_names,
            columns=adata.var_names,
        )
    else:
        expr = pd.DataFrame(
            adata.X,
            index=adata.obs_names,
            columns=adata.var_names,
        )

    # Add grouping variable
    expr[groupby] = adata.obs[groupby].values

    # ----------------------------
    # Get signature matrix
    # ----------------------------
    signature = expr.groupby(groupby).mean().T

    # ----------------------------
    # 6. Save if requested
    # ----------------------------
    if output_path is not None:
        signature.to_csv(output_path)
        print(f" Signature matrix saved to: {output_path}")

    return signature
    