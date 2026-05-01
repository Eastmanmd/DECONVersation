# ===============================
# Import Libraries
# ===============================
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import scipy
import scipy.sparse as sp 
import anndata
from anndata import AnnData
from sklearn.utils import check_random_state
import warnings

def generate_pseudobulk(
    adata: ad.AnnData,
    cell_type_col: str, 
    n_pseudobulks: int = 1000, 
    target_proportion_min: float = 0.1,
    target_proportion_max: float = 0.80, 
    n_cells_per_pseudobulk: int = 600, 
    random_state: int = None,  
    min_cells_threshold: float = 0.1
):
    """
    Create pseudobulk samples with controlled cell type proportions.
    
    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix
    cell_type_col : str
        Column name in adata.obs containing cell type labels
    n_pseudobulks : int
        Number of pseudobulk samples to generate
    target_proportion_min : float
        Minimum proportion for target cell type
    target_proportion_max : float
        Maximum proportion for target cell type
    n_cells_per_pseudobulk : int
        Number of cells to sample per pseudobulk
    random_state : int
        Random seed for reproducibility
    min_cells_threshold : float
        Warn if available cells < this fraction of needed cells
    
    Returns
    -------
    pseudobulk_df : pd.DataFrame
        Expression matrix (pseudobulk sample × genes)
    proportions_df : pd.DataFrame
        Cell type proportions (pseudobulk sample  × cell types)
    """
    
    # Set random state for reproducibility (optional)
    rng = check_random_state(random_state)


    # validate cell type column presence in anndata
    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"Column '{col}' not found in adata.obs.")
    
    # Get unique cell types
    all_cell_types = adata.obs[cell_type_col].unique()
    
    # Output empty dataframe if no cells 
    if len(all_cell_types) == 0:
        warnings.warn("No cell types found; returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Group cell tyoes
    grouped_cells = {k: v.index.tolist() for k, v in adata.obs.groupby(cell_type_col, observed=True)}

    # Initialize empty lists to store results
    pseudobulk_data_list, proportions_data_list, sample_names = [], [], []
    
    # Generate random target proportions
    target_proportions = rng.uniform(
        low=target_proportion_min, 
        high=target_proportion_max, 
        size=n_pseudobulks
    )

    # Iterate through pseudobulk sample
    for i in range(n_pseudobulks):
        
        # Randomly sample cell types
        target_type = rng.choice(all_cell_types)

        # Select target proportion
        current_prop = target_proportions[i]
        
        # Calculate number of target and other cells needed
        n_target = int(np.round(n_cells_per_pseudobulk * current_prop))
        n_target = max(0, min(n_target, n_cells_per_pseudobulk))
        
        # number of other cells needed to complete proportion
        n_other = n_cells_per_pseudobulk - n_target 

        # list to store selected cell indices
        selected_indices = []
        
        # Sample target cells
        if target_type in grouped_cells and n_target > 0: 

            # Number of cells availale for subsampling
            available_target = len(grouped_cells[target_type])
            
            # Warning for small cell populations (# Only used if the number of target cells is greater 
            #than number of available cells to be sampled
            if available_target < n_target * min_cells_threshold:
                warnings.warn(
                    f"Pseudobulk {i+1}: Cell type '{target_type}' has only "
                    f"{available_target} cells but needs {n_target}. "
                    f"Using replacement sampling."
                )
            
                target_cells = rng.choice(
                    grouped_cells[target_type], 
                    size=n_target, 
                    replace=True
                ).tolist()
                selected_indices.extend(target_cells)
                print("Sampling with replacement as not a lot of available cells")

            # Without replacement sampling
            else:
                target_cells = rng.choice(
                    grouped_cells[target_type], 
                    size=n_target, 
                    replace=False
                    ).tolist()
                
                selected_indices.extend(target_cells)
        
        # Sample other cells
        if n_other > 0:

            # Exclude target cell type
            other_types = [ct for ct in all_cell_types if ct != target_type] 

            # Get the indices of all other cell types excluding target cell
            all_other_indices = sum((grouped_cells.get(ct, []) for ct in other_types), [])
            
            # Handle case where other cells are exhausted (fill with target cells)
            if len(all_other_indices) == 0:
                warnings.warn(
                    f"Pseudobulk {i+1}: No other cell types available. "
                    f"Filling with additional '{target_type}' cells."
                )
                if target_type in grouped_cells:
                    other_cells = rng.choice(
                        grouped_cells[target_type], 
                        size=n_other, 
                        replace=False
                    ).tolist()
                    selected_indices.extend(other_cells)
                    
            else:
                other_cells = rng.choice(
                    all_other_indices, 
                    size=n_other, 
                    replace=False
                ).tolist()
                selected_indices.extend(other_cells)
        
        # Skip if no cells were selected
        if not selected_indices:
            warnings.warn(f"Pseudobulk {i+1}: No cells selected, skipping.")
            continue
        
        # Sum expression
        pb_matrix = adata[selected_indices, :].X
        pb_sum = pb_matrix.sum(axis=0).A1 if sp.issparse(pb_matrix) else pb_matrix.sum(axis=0)
        pseudobulk_data_list.append(pb_sum)
        
        # Compute proportions that match the expression aggregation
        # Count occurrences including duplicates from replacement sampling
        selected_types = adata.obs.loc[selected_indices, cell_type_col]
        type_counts = selected_types.value_counts()
        total_cells = len(selected_indices)
        
        # Create proportion dictionary with all cell types
        props = {ct: type_counts.get(ct, 0) / total_cells for ct in all_cell_types}
        proportions_data_list.append(props)
        
        sample_names.append(f"{target_type}_PB{i+1}")
            
    # Handle case where no pseudobulks were created
    if not pseudobulk_data_list:
        warnings.warn("No pseudobulks were created; returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Create output DataFrames
    pseudobulk_df = pd.DataFrame(
        pseudobulk_data_list, 
        index=sample_names, 
        columns=adata.var_names
    )
    proportions_df = pd.DataFrame(proportions_data_list, index=sample_names)
    
    return pseudobulk_df, proportions_df


