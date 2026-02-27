#!/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python

"""
Example usage:
--------------
    
python create_proportional_pseudobulks.py \
  --adata adata_sub.h5ad \
  --cell_type_col heca_lineage \
  --id_col id \
  --n_pseudobulks 400 \
  --n_cells_per_pseudobulk 600 \
  --target_proportion_min 0.4 \
  --target_proportion_max 0.85 \
  --out_prefix czi_pb
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from sklearn.utils import check_random_state

def create_proportional_pseudobulks(
    adata: ad.AnnData,
    cell_type_col: str,
    n_pseudobulks: int = 1000,
    target_proportion_min: float = 0.1,
    target_proportion_max: float = 0.80,
    n_cells_per_pseudobulk: int = 600,
    random_state: int = None,
    min_cells_threshold: float = 0.1
):

    rng = check_random_state(random_state)
    all_cell_types = adata.obs[cell_type_col].unique()

    if len(all_cell_types) == 0:
        warnings.warn("No cell types found; returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    grouped_cells = {k: v.index.tolist() for k, v in adata.obs.groupby(cell_type_col, observed=True)}

    pseudobulk_data_list, proportions_data_list, sample_names = [], [], []

    target_proportions = rng.uniform(
        low=target_proportion_min,
        high=target_proportion_max,
        size=n_pseudobulks
    )

    for i in range(n_pseudobulks):

        target_type = rng.choice(all_cell_types)
        current_prop = target_proportions[i]

        n_target = int(np.round(n_cells_per_pseudobulk * current_prop))
        n_target = max(0, min(n_target, n_cells_per_pseudobulk))
        n_other = n_cells_per_pseudobulk - n_target

        selected_indices = []

        if target_type in grouped_cells and n_target > 0:
            available_target = len(grouped_cells[target_type])

            if available_target < n_target * min_cells_threshold:
                warnings.warn(
                    f"Pseudobulk {i+1}: Cell type '{target_type}' has only "
                    f"{available_target} cells but needs {n_target}. Using replacement."
                )
                target_cells = rng.choice(grouped_cells[target_type], size=n_target, replace=True).tolist()
            else:
                target_cells = rng.choice(grouped_cells[target_type], size=n_target, replace=False).tolist()

            selected_indices.extend(target_cells)

        if n_other > 0:
            other_types = [ct for ct in all_cell_types if ct != target_type]
            all_other_indices = sum((grouped_cells.get(ct, []) for ct in other_types), [])

            if len(all_other_indices) == 0:
                warnings.warn(f"Pseudobulk {i+1}: No other cell types available.")
                other_cells = rng.choice(grouped_cells[target_type], size=n_other, replace=False).tolist()
            else:
                other_cells = rng.choice(all_other_indices, size=n_other, replace=False).tolist()

            selected_indices.extend(other_cells)

        if not selected_indices:
            warnings.warn(f"Pseudobulk {i+1}: No cells selected, skipping.")
            continue

        pb_matrix = adata[selected_indices, :].X
        pb_sum = pb_matrix.sum(axis=0).A1 if sp.issparse(pb_matrix) else pb_matrix.sum(axis=0)
        pseudobulk_data_list.append(pb_sum)

        selected_types = adata.obs.loc[selected_indices, cell_type_col]
        type_counts = selected_types.value_counts()
        total_cells = len(selected_indices)

        props = {ct: type_counts.get(ct, 0) / total_cells for ct in all_cell_types}
        proportions_data_list.append(props)

        sample_names.append(f"{target_type}_PB{i+1}")

    if not pseudobulk_data_list:
        warnings.warn("No pseudobulks were created.")
        return pd.DataFrame(), pd.DataFrame()

    pseudobulk_df = pd.DataFrame(pseudobulk_data_list, index=sample_names, columns=adata.var_names)
    proportions_df = pd.DataFrame(proportions_data_list, index=sample_names)

    return pseudobulk_df, proportions_df


def main():

    parser = argparse.ArgumentParser(description="Create proportional pseudobulks from AnnData.")

    parser.add_argument("--adata", required=True, help="Input AnnData (.h5ad)")
    parser.add_argument("--cell_type_col", required=True, help="Cell type column in adata.obs")
    parser.add_argument("--id_col", default="id", help="Sample ID column in adata.obs")

    parser.add_argument("--n_pseudobulks", type=int, default=400)
    parser.add_argument("--n_cells_per_pseudobulk", type=int, default=600)
    parser.add_argument("--target_proportion_min", type=float, default=0.4)
    parser.add_argument("--target_proportion_max", type=float, default=0.85)
    parser.add_argument("--random_state", type=int, default=None)

    parser.add_argument("--out_prefix", required=True, help="Output prefix")

    args = parser.parse_args()

    print("Loading:", args.adata)
    adata = sc.read_h5ad(args.adata)

    all_pseudobulks = []
    all_cell_props = []

    id_list = adata.obs[args.id_col].unique()

    for sample_id in id_list:

        adata_subset = adata[adata.obs[args.id_col] == sample_id].copy()

        if adata_subset.n_obs == 0:
            print(f"No cells found for ID: {sample_id}")
            continue

        pb_expr, pb_prop = create_proportional_pseudobulks(
            adata=adata_subset,
            cell_type_col=args.cell_type_col,
            target_proportion_min=args.target_proportion_min,
            target_proportion_max=args.target_proportion_max,
            n_cells_per_pseudobulk=args.n_cells_per_pseudobulk,
            n_pseudobulks=args.n_pseudobulks,
            random_state=args.random_state,
        )

        all_pseudobulks.append(pb_expr)
        all_cell_props.append(pb_prop)

    pseudobulk_czi = pd.concat(all_pseudobulks, axis=0, ignore_index=True)
    pseudobulk_czi_cell_prop = pd.concat(all_cell_props, axis=0, ignore_index=True)

    expr_out = f"{args.out_prefix}.pseudobulk_expr.tsv"
    prop_out = f"{args.out_prefix}.pseudobulk_props.tsv"

    pseudobulk_czi.to_csv(expr_out, sep="\t")
    pseudobulk_czi_cell_prop.to_csv(prop_out, sep="\t")

    print("Saved:")
    print(expr_out)
    print(prop_out)


if __name__ == "__main__":
    main()


