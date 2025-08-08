# The goal of this script is to select the set of parameters that produces 
# the best model results without over-fitting the models
# Main hyper-parameters to tune

# In the radom forest regressor
# "n_estimators"
# "max_depth"

# Parameters to tune in the pseudobulk model
# n_cells_per_pseudobulk 


# Load libraries 
import pandas as pd
import numpy as np
from sklearn.utils import check_random_state
import warnings

# Import libraries
import pickle
import numpy as np
import anndata
import anndata as ad
import pandas as pd
import scanpy as sc
import scipy.sparse as sp 
import matplotlib.pyplot as plt
import anndata
from anndata import AnnData
import warnings

# For plotting
import seaborn as sns

# For PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# For umap
import umap

# SCVI
import scvi
import cellxgene_census
import cellxgene_census.experimental

# For ML training
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from scipy.sparse import issparse, vstack

# ------------------------------------------- Load Anndata ----------------------------------------------------------
# First read in anndata
czi_path = "/gpfs/commons/groups/compbio/projects/CZI_endom/RNA_temp/62_harmony_102224_Seuratv34_newannot_counts.h5ad"
adata = sc.read_h5ad(czi_path)

adata.layers["counts"] = adata.raw.X.copy()
adata.layers["logcounts"] = adata.X.copy()

# Add HECA annotations to adata (annotation from published HECA dataset)
heca_annot = pd.read_csv("/nfs/home/rfu/projects/CZI_endom/CZI_HECAtype.csv.gz")
adata.obs["heca_celltype"] = heca_annot["celltype"].values
adata.obs["heca_lineage"] = heca_annot["lineage"].values

#Get ensembl IDs - map gene symbol to ensembl ID
gene_ids = pd.read_csv("/gpfs/commons/groups/compbio/projects/CZI_endom/RNA_temp/gene_names_gene_ids_czi_endo.csv", index_col= False)

gene_id_dict = pd.DataFrame({
    'gene_symbol': gene_ids["gene_name"],
    'ensembl_id': gene_ids["gene_id"]})

gene_id_dict = gene_id_dict.set_index('gene_symbol')['ensembl_id'].to_dict()

ensembl_ids = [gene_id_dict.get(gene, 'NA') for gene in adata.var_names]

# Add ensembl IDs to anndata object
adata.var["gene_names"] = adata.var_names

adata.var["ensembl_ids"] = ensembl_ids
adata.var_names = ensembl_ids

# Remove duplicated IDs
adata = adata[:, ~adata.var["ensembl_ids"].duplicated()].copy()

# Check for duplicated IDs
adata.var["ensembl_ids"].duplicated().any()

del adata.varm["HARMONY"]
del adata.varm["PCs"]

# Split data into training and testing sets
# Get unique sample IDs
unique_ids = adata.obs['id'].unique()

# Split the sample IDs
train_ids, test_ids = train_test_split(unique_ids, test_size=0.15, random_state=42)

# Create boolean masks
train_mask = adata.obs['id'].isin(train_ids)
test_mask = adata.obs['id'].isin(test_ids)

# Subset the AnnData object
adata_train = adata[train_mask].copy()
adata_test = adata[test_mask].copy()

#------------------------------------------ Define pseudobulk function --------------------------------------------------

def create_proportional_pseudobulks(
    adata: anndata.AnnData,  # anndata object from a single sample
    cell_type_col: str,      # column name corresponding to the cell type
    n_pseudobulks: int = 1000,
    target_proportion_min: float = 0.1,
    target_proportion_max: float = 0.9,
    n_cells_per_pseudobulk: int = 800, # Fixed number of cells per pseudobulk
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # --- Input Validation ---
    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"Cell type column '{cell_type_col}' not found in adata.obs.")
    if not (0 <= target_proportion_min < target_proportion_max <= 1):
        raise ValueError("target_proportion_min must be less than target_proportion_max, "
                         "and both must be between 0 and 1.")
    if not (n_cells_per_pseudobulk > 0):
        raise ValueError("n_cells_per_pseudobulk must be greater than 0.")

    # --- Setup ---
    rng = check_random_state(random_state)
    all_cell_types = adata.obs[cell_type_col].unique()
    
    # Check if there are at least two cell types to allow for "other" cells
    if len(all_cell_types) < 2 and (target_proportion_max < 1.0 or target_proportion_min > 0.0):
        warnings.warn("Less than 2 unique cell types found. Proportional mixing might be limited "
                      "if target_proportion_min is not 0.0 or target_proportion_max is not 1.0.")
    elif len(all_cell_types) < 1:
        warnings.warn("No cell types found in adata.obs. Returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    pseudobulk_data_list = []
    proportions_data_list = []
    pseudobulk_sample_names = []

    # Generate random target proportions
    target_proportions = rng.uniform(low=target_proportion_min, high=target_proportion_max, size=n_pseudobulks)
    
    # Cycle through cell types to be the 'target'
    cell_type_cycle = np.tile(all_cell_types, int(np.ceil(n_pseudobulks / len(all_cell_types))))[:n_pseudobulks]

    # Group cells by cell type for efficient sampling
    grouped_cells = adata.obs.groupby(cell_type_col).apply(lambda x: x.index.tolist()).to_dict()

    # --- Create multiple pseudobulks based on controlled proportions ---
    for i in range(n_pseudobulks):
        target_cell_type = cell_type_cycle[i]
        current_target_prop = target_proportions[i]

        # Calculate cell counts for target and other cells
        n_target_cells = int(np.round(n_cells_per_pseudobulk * current_target_prop))
        n_target_cells = max(0, min(n_target_cells, n_cells_per_pseudobulk)) # Ensure within bounds

        n_other_cells = n_cells_per_pseudobulk - n_target_cells

        selected_indices = []

        # 1. Sample Target Cell Type
        if target_cell_type in grouped_cells and len(grouped_cells[target_cell_type]) > 0:
            if n_target_cells > 0:
                target_cell_indices = rng.choice(
                    grouped_cells[target_cell_type],
                    size=n_target_cells,
                    replace=True # Allow replacement to hit exact proportions, even if few cells
                ).tolist()
                selected_indices.extend(target_cell_indices)
        elif n_target_cells > 0:
            warnings.warn(f"Cell type '{target_cell_type}' selected as target for PB_{i+1}, "
                          f"but has no cells. This pseudobulk will have 0% of '{target_cell_type}'.")
            # This pseudobulk will just consist of 'other' cells if target has no cells.

        # 2. Sample "Other" Cell Types
        if n_other_cells > 0:
            other_cell_types = [ct for ct in all_cell_types if ct != target_cell_type]
            
            if len(other_cell_types) > 0:
                # Get all indices of 'other' cell types
                all_other_indices = []
                for ct in other_cell_types:
                    if ct in grouped_cells and len(grouped_cells[ct]) > 0:
                        all_other_indices.extend(grouped_cells[ct])
                
                if len(all_other_indices) > 0:
                    other_cell_selected_indices = rng.choice(
                        all_other_indices,
                        size=n_other_cells,
                        replace=True # Allow replacement for 'other' cells
                    ).tolist()
                    selected_indices.extend(other_cell_selected_indices)
                else:
                    warnings.warn(f"No 'other' cell types available for sampling for PB_{i+1}. "
                                  f"This pseudobulk might consist solely of the target type or be empty.")
            elif n_target_cells == 0: # If no other cell types and no target cells selected
                 warnings.warn(f"No cells available to sample for PB_{i+1}. Skipping this pseudobulk.")
                 continue # Skip to next pseudobulk if no cells can be sampled

        # Handle cases where selected_indices might be empty (e.g., if n_cells_per_pseudobulk is 0 or no cells available)
        if not selected_indices:
            warnings.warn(f"No cells sampled for pseudobulk PB_{i+1}. Skipping this pseudobulk.")
            continue

        # Store the pseudobulk expression
        pb_expression_matrix = adata[selected_indices, :].X
        if isinstance(pb_expression_matrix, (np.ndarray, pd.DataFrame)):
            pb_expression_sum = pb_expression_matrix.sum(axis=0)
        else: # Assume sparse matrix (e.g., scipy.sparse.csr_matrix)
            pb_expression_sum = pb_expression_matrix.sum(axis=0).A1 

        pseudobulk_data_list.append(pb_expression_sum)

        # Calculate and store the actual cell type proportions
        selected_cell_types_series = adata.obs.loc[selected_indices, cell_type_col]
        current_pb_counts = selected_cell_types_series.value_counts()
        current_pb_proportions = current_pb_counts / current_pb_counts.sum()

        proportions_dict = {ct: 0.0 for ct in all_cell_types}
        for ct, prop in current_pb_proportions.items():
            proportions_dict[ct] = prop
        proportions_data_list.append(proportions_dict)

        # Generate a unique name for the pseudobulk sample
        pseudobulk_sample_names.append(f"{target_cell_type}_PB{i+1}")
                
    # --- Create final DataFrames ---
    if not pseudobulk_data_list:
        warnings.warn("No pseudobulk samples were successfully created. Returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    pseudobulk_df = pd.DataFrame(
        pseudobulk_data_list,
        index=pseudobulk_sample_names,
        columns=adata.var_names
    )

    proportions_df = pd.DataFrame(
        proportions_data_list,
        index=pseudobulk_sample_names
    )
    # Ensure all_cell_types are columns, filling missing with 0 for consistency
    proportions_df = proportions_df.reindex(columns=all_cell_types, fill_value=0.0)

    return pseudobulk_df, proportions_df

    
# ----------------------------------------- Get Embeddings --------------------------------------------------------------