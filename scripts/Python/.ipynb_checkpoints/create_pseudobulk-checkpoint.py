"""
Create pseudobulk profiles from single-cell data.

This script:
1. Aggregates single-cell data into pseudobulk samples by grouping on a specified variable (e.g., sample ID).
2. Computes cell-type proportions per pseudobulk.
3. Saves both pseudobulk data and cell-type proportions to CSV files.

Example usage:
---------------
python create_pseudobulk.py \
    --adata_path /path/to/single_cell_data.h5ad \
    --group_var id \
    --celltype_var heca_lineage \
    --n_cells_per_pseudobulk 800 \
    --out_pseudobulk_path /path/to/output/pseudobulk.csv \
    --out_prop_path /path/to/output/cell_proportions.csv
"""

# Import libraries
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import anndata as ad
import psutil
import logging
from sklearn.utils import check_random_state
import argparse

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------- #
# Pseudobulk creation function  #
# ----------------------------- #
def create_proportional_pseudobulks(
    adata: ad.AnnData,
    cell_type_col: str,
    n_pseudobulks: int = 1000,
    target_proportion_min: float = 0.1,
    target_proportion_max: float = 0.9,
    n_cells_per_pseudobulk: int = 800,
    random_state: int = 42
):
    # Set random state for reproducibilty
    rng = check_random_state(random_state)

    # Get unique cell types
    all_cell_types = adata.obs[cell_type_col].unique()

    # Output empty dataframe if no cells 
    if len(all_cell_types) == 0:
        warnings.warn("No cell types found; returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    grouped_cells = adata.obs.groupby(cell_type_col).apply(lambda x: x.index.tolist()).to_dict()
    pseudobulk_data_list, proportions_data_list, sample_names = [], [], []

    target_proportions = rng.uniform(low=target_proportion_min, high=target_proportion_max, size=n_pseudobulks)
    cell_type_cycle = np.tile(all_cell_types, int(np.ceil(n_pseudobulks / len(all_cell_types))))[:n_pseudobulks]

    for i in range(n_pseudobulks):
        target_type = cell_type_cycle[i]
        current_prop = target_proportions[i]
        n_target = int(np.round(n_cells_per_pseudobulk * current_prop))
        n_target = max(0, min(n_target, n_cells_per_pseudobulk))
        n_other = n_cells_per_pseudobulk - n_target
        selected_indices = []

        # Sample target cells
        if target_type in grouped_cells and n_target > 0:
            target_cells = rng.choice(grouped_cells[target_type], size=n_target, replace=True).tolist()
            selected_indices.extend(target_cells)

        # Sample other cells
        if n_other > 0:
            other_types = [ct for ct in all_cell_types if ct != target_type]
            all_other_indices = sum((grouped_cells.get(ct, []) for ct in other_types), [])
            if all_other_indices:
                other_cells = rng.choice(all_other_indices, size=n_other, replace=True).tolist()
                selected_indices.extend(other_cells)

        if not selected_indices:
            continue

        # Sum expression
        pb_matrix = adata[selected_indices, :].X
        pb_sum = pb_matrix.sum(axis=0).A1 if sp.issparse(pb_matrix) else pb_matrix.sum(axis=0)
        pseudobulk_data_list.append(pb_sum)

        # Compute proportions
        selected_types = adata.obs.loc[selected_indices, cell_type_col]
        props = selected_types.value_counts(normalize=True).to_dict()
        proportions_data_list.append({ct: props.get(ct, 0.0) for ct in all_cell_types})

        sample_names.append(f"{target_type}_PB{i+1}")

    if not pseudobulk_data_list:
        return pd.DataFrame(), pd.DataFrame()

    pseudobulk_df = pd.DataFrame(pseudobulk_data_list, index=sample_names, columns=adata.var_names)
    proportions_df = pd.DataFrame(proportions_data_list, index=sample_names)
    return pseudobulk_df, proportions_df


# ----------------------------------- #
# Oversample minority class function  #
# ----------------------------------- #
def oversample_celltypes(
    adata,
    group_col="heca_lineage",
    target_n=500,
    random_state=42
):
    """
    Oversample minority cell types in an AnnData object so that each group 
    has at least `target_n` cells, and ensure unique cell indices.
    """
    np.random.seed(random_state)
    
    obs = adata.obs.copy()
    group_counts = obs[group_col].value_counts()
    
    oversampled_indices = []
    
    for group, count in group_counts.items():
        group_idx = obs.index[obs[group_col] == group]
        if count < target_n:
            # sample with replacement
            sampled_idx = np.random.choice(group_idx, size=target_n, replace=True)
        else:
            sampled_idx = group_idx
        oversampled_indices.extend(sampled_idx)
    
    adata_balanced = adata[oversampled_indices].copy()
    
    # assign unique cell names
    new_obs_names = [
        f"{name}_oversampled_{i}" for i, name in enumerate(adata_balanced.obs_names)
    ]
    adata_balanced.obs_names = new_obs_names
    
    shuffled = np.random.permutation(adata_balanced.n_obs)
    adata_balanced = adata_balanced[shuffled].copy()
    
    return adata_balanced
    


# ----------------------------- #
# Main script logic             #
# ----------------------------- #
def main(args):

    # Set logger
    logging.basicConfig(format='%(asctime)s %(message)s', 
                        filename=args.log_file,
                        filemode='w', force = True)

    # Log: load AnnData Object
    logging.warning('Loading AnnData Object')

    # Read single cell data (h5ad object)
    adata = sc.read_h5ad(args.adata_path)

    logging.warning('Done: Loading AnnData Object')

    # Check if sample ID column is present indata
    if args.group_var not in adata.obs.columns:
        raise ValueError(f"Group variable '{args.group_var}' not found in adata.obs.")
    if args.celltype_var not in adata.obs.columns:
        raise ValueError(f"Cell-type variable '{args.celltype_var}' not found in adata.obs.")

    # Subset to specific sample IDs if provided
    logging.warning('Subsetting adata to provided sample IDs')
    if args.sample_ids:
        print(f"Subsetting to provided sample IDs: {args.sample_ids}")
        adata = adata[adata.obs[args.group_var].isin(args.sample_ids)].copy()

        # Raise error if no cells remaining after subsetting to select sample IDs 
        if adata.n_obs == 0:
            raise ValueError("No cells remain after subsetting to the specified sample IDs.")

        # Get sample IDs
        sample_ids = adata.obs[args.group_var].unique().tolist()

        if args.combine_samples:
            adata.obs["_combined_sample"] = "global_sample"
            args.group_var = "_combined_sample"
            sample_ids = ["global_sample"]
    else:
        # Treat entire adata as one combined sample
        print("No sample IDs provided; treating all cells as one combined sample.")

        # Add new obs variable with a single ID name
        adata.obs["_combined_sample"] = "global_sample"

        # Reset group variable (sample ID) as global sample ID
        args.group_var = "_combined_sample"
        sample_ids = ["global_sample"]

    logging.warning('Subsetting Done')

    logging.warning('Generating pseudobulk data')

    # Initialize empty lists to contain pseudobulks and proportions
    all_pseudobulks, all_props = [], []

    # Iterate through samples
    for sid in sample_ids:
        
        print(f"  Processing {sid}...")
        subset = adata[adata.obs[args.group_var] == sid].copy()

        # Skip if no cells are found for specified sample
        if subset.n_obs == 0:
            print(f"    No cells found for {sid}, skipping.")
            continue

        # Create pseudobulk 
        pb, props = create_proportional_pseudobulks(
            adata=subset,
            n_pseudobulks= args.n_pseudobulks,
            cell_type_col=args.celltype_var,
            n_cells_per_pseudobulk=args.n_cells_per_pseudobulk,
            random_state=args.random_state,
        )
        pb.index = [f"{sid}_{i}" for i in pb.index]
        props.index = pb.index

        all_pseudobulks.append(pb)
        all_props.append(props)

    logging.warning('Generating pseudobulk done')

    print("Concatenating results...")
    pseudobulk_df = pd.concat(all_pseudobulks, axis=0)
    props_df = pd.concat(all_props, axis=0)

    logging.warning('Saving pseudobulk and cell props')
    print(f"Saving outputs:\n - Pseudobulk data → {args.out_pseudobulk_path}\n - Cell proportions → {args.out_prop_path}")
    pseudobulk_df.to_csv(args.out_pseudobulk_path)
    
    props_df.to_csv(args.out_prop_path)
    print("Completed generating pseudobulk data and cell-type proportions.")

    logging.warning('Pseudobulk saved')


# ----------------------------- #
# Argument parser               #
# ----------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudobulk datasets and cell-type proportions.")
    parser.add_argument("--adata_path", required=True, help="Path to input AnnData (.h5ad) file")
    parser.add_argument("--group_var", required=True, help="Column in adata.obs to group by (e.g., sample ID)")
    parser.add_argument("--celltype_var", required=True, help="Column in adata.obs defining cell types")
    parser.add_argument("--sample_ids", nargs="+", default=None, help="Optional list of sample IDs to subset and process")
    parser.add_argument("--n_cells_per_pseudobulk", type=int, default=800, help="Number of cells per pseudobulk")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--n_pseudobulks", type=int, default=1000, help="Random seed for reproducibility")
    parser.add_argument("--out_pseudobulk_path", required=True, help="Output CSV path for pseudobulk data")
    parser.add_argument("--out_prop_path", required=True, help="Output CSV path for cell-type proportions")
    parser.add_argument("--log_file", default="./logs/logLfile.log", help="Log output file")
    parser.add_argument("--combine_samples", type=bool, default=False, help="Create pseudobulk by across all samples")
    
    args = parser.parse_args()
    main(args)