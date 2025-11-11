"""
Create pseudobulk profiles from single-cell data and extract Geneformer embeddings.

This script:
1. Aggregates single-cell data into pseudobulk samples.
2. Computes cell-type proportions per pseudobulk.
3. Generates Geneformer embeddings from the pseudobulked data.

Example usage:
---------------
python create_pseudobulk_extract_embed_gf.py \
    --adata_path /path/to/single_cell_data.h5ad \
    --heca_path /path/to/HECA/annotations/heca_celltypes.csv.gz \
    --gene_map_path /path/to/gene_names_ensembl_ids.csv \
    --sample_ids sample1 sample2 sample3 \
    --out_prop_path /path/to/output/pseudobulk_cell_prop_test.csv \
    --out_adata_path /path/to/output/pseudobulk_test.h5ad \
    --token_output_dir /path/to/output/tokenized_data/ \
    --token_output_name token_train_set \
    --embedding_output_path /path/to/output/gf_zeroshot_pseudobulk_train_set_test.csv
"""

import warnings
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
from anndata import AnnData
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import umap
import cellxgene_census
import cellxgene_census.experimental
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from scipy.sparse import issparse, vstack
from geneformer import EmbExtractor, TranscriptomeTokenizer, get_embs
from geneformer import perturber_utils as pu
import argparse
import os

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------- #
# Pseudobulk creation function  #
# ----------------------------- #
# Cleaner function but exactly the same as the above function
def create_proportional_pseudobulks(
    adata: ad.AnnData,
    cell_type_col: str,
    n_pseudobulks: int = 1000,
    target_proportion_min: float = 0.1,
    target_proportion_max: float = 0.9,
    n_cells_per_pseudobulk: int = 800,
    random_state: int = 42
):
    rng = check_random_state(random_state)
    all_cell_types = adata.obs[cell_type_col].unique()

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
# oversample minority class function  #
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
            # if already has more than target, just take all (no undersampling)
            sampled_idx = group_idx
        oversampled_indices.extend(sampled_idx)
    
    # subset adata
    adata_balanced = adata[oversampled_indices].copy()
    
    # assign unique cell names
    new_obs_names = [
        f"{name}_oversampled_{i}" for i, name in enumerate(adata_balanced.obs_names)
    ]
    adata_balanced.obs_names = new_obs_names
    
    # optionally shuffle
    shuffled = np.random.permutation(adata_balanced.n_obs)
    adata_balanced = adata_balanced[shuffled].copy()
    
    return adata_balanced


# ----------------------------- #
# Main script logic             #
# ----------------------------- #
def main(args):
    print("Loading AnnData object...")
    adata = sc.read_h5ad(args.adata_path)

    print("Preparing data layers...")
    adata.layers["counts"] = adata.raw.X.copy()
    adata.layers["logcounts"] = adata.X.copy()

    print("Adding HECA annotations...")
    heca_annot = pd.read_csv(args.heca_path)
    adata.obs["heca_celltype"] = heca_annot["celltype"].values
    adata.obs["heca_lineage"] = heca_annot["lineage"].values

    print("Mapping gene symbols to Ensembl IDs...")
    gene_ids = pd.read_csv(args.gene_map_path)
    mapping = dict(zip(gene_ids["gene_name"], gene_ids["gene_id"]))
    adata.var["gene_names"] = adata.var_names
    adata.var["ensembl_id"] = [mapping.get(g, "NA") for g in adata.var_names]
    adata = adata[:, ~adata.var["ensembl_id"].duplicated()].copy()
    adata.var_names = adata.var["ensembl_id"]

    adata.obs["n_counts"] = np.array(adata.X.sum(axis=1)).flatten()

    # Only limit to a few datasets
    #adata = adata[adata.obs["heca_lineage"].isin(["EpithelialGL_harmonal", "Endothelial", "Stromal_EMT", "EpithelialPino"])]

    print("Splitting into train/test sets...")
    #train_ids, test_ids = train_test_split(adata.obs["id"].unique(), test_size=0.15, random_state=42)
    #adata_train = adata[adata.obs["id"].isin(train_ids)].copy()

    print("Generating pseudobulk data...")
    all_pseudobulks, all_props = [], []
    for sid in args.sample_ids:
        print(f"  Processing {sid}...")
        subset = adata[adata.obs["id"] == sid].copy()
        #subset = oversample_celltypes(subset, target_n=200) # oversampling minority class
        
        if subset.n_obs == 0:
            print(f" No cells found for {sid}, skipping.")
            continue

        pb, props = create_proportional_pseudobulks(
            adata=subset,
            cell_type_col="heca_lineage",
            n_cells_per_pseudobulk=args.n_cells_per_pseudobulk,
            random_state=args.random_state,
        )
        all_pseudobulks.append(pb)
        all_props.append(props)

    print("Concatenating results...")
    #pseudobulk_df = pd.concat(all_pseudobulks, axis=0, ignore_index=True)
    #props_df = pd.concat(all_props, axis=0, ignore_index=True)
    if len(all_pseudobulks) > 1:
        pseudobulk_df = pd.concat(all_pseudobulks, axis=0, ignore_index=True)
    else:
        pseudobulk_df = all_pseudobulks[0] if all_pseudobulks else pd.DataFrame()
    
    if len(all_props) > 1:
        props_df = pd.concat(all_props, axis=0, ignore_index=True)
    else:
        props_df = all_props[0] if all_props else pd.DataFrame()

    print(f"Saving outputs:\n - Cell proportions → {args.out_prop_path}\n - Pseudobulk AnnData → {args.out_adata_path}")
    
    pseudobulk_df.to_csv(f"{args.out_prop_path}pseudobulk_raw.csv") # save raw pseudobulk data
    props_df.to_csv(f"{args.out_prop_path}cell_prop.csv") # save proportions data

    #props_df.to_csv(args.out_prop_path) # save proportions

    pb_adata = sc.AnnData(pseudobulk_df)
    pb_adata.obs["cell_type"] = "unknown"
    pb_adata.obs["n_counts"] = np.sum(pb_adata.X, axis=1).tolist()
    pb_adata.var["ensembl_id"] = pb_adata.var_names
    pb_adata.X = sp.csc_matrix(pb_adata.X)
    pb_adata.write_h5ad(args.out_adata_path)

    print("Completed generating pseudobulk data and cell type proportions.")

    # ----------------------------- #
    # Geneformer tokenization       #
    # ----------------------------- #
    print("\nStarting Geneformer tokenization and embedding extraction...")

    tk = TranscriptomeTokenizer(
        {"cell_type": "cell_type"},
        model_input_size=4096,
        special_token=True,
        chunk_size=512,
        gene_median_file=args.gene_median_file,
        token_dictionary_file=args.token_dictionary_file,
        gene_mapping_file=args.gene_mapping_file
    )

    tk.tokenize_data(
        os.path.dirname(args.out_adata_path),
        args.token_output_dir,
        args.token_output_name,
        file_format="h5ad"
    )

    model = pu.load_model(
        model_type="Pretrained",
        num_classes=0,
        model_directory=args.geneformer_model_path,
        mode="eval"
    )

    with open(args.token_dictionary_file, "rb") as f:
        gene_token_dict = pickle.load(f)
    token_gene_dict = {v: k for k, v in gene_token_dict.items()}
    pad_token_id = gene_token_dict.get("<pad>")

    filtered_input_data = pu.load_and_filter(
        filter_data=None,
        nproc=1,
        input_data_file=f"{args.token_output_dir}/{args.token_output_name}.dataset/"
    )

    state_embs_dict = get_embs(
        model,
        filtered_input_data,
        emb_mode="cell",
        layer_to_quant=0,
        pad_token_id=pad_token_id,
        token_gene_dict=token_gene_dict,
        special_token=True,
        forward_batch_size=50
    )

    tempout = pd.DataFrame.from_dict(state_embs_dict.cpu().numpy())
    df = filtered_input_data.to_pandas()
    tempout.index = df['cell_type'] + '_' + df.index.astype(str)
    tempout = tempout.reset_index(drop=True)
    tempout.to_csv(args.embedding_output_path)

    print(f"Geneformer embeddings saved to: {args.embedding_output_path}")


# ----------------------------- #
# Argument parser               #
# ----------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudobulk datasets and Geneformer embeddings.")
    parser.add_argument("--adata_path", required=True, help="Path to input AnnData (.h5ad) file")
    parser.add_argument("--heca_path", required=True, help="Path to HECA annotation CSV file")
    parser.add_argument("--gene_map_path", required=True, help="Path to gene symbol-to-Ensembl ID mapping CSV")
    parser.add_argument("--sample_ids", nargs="+", required=True, help="List of sample IDs to process")
    parser.add_argument("--out_prop_path", required=True, help="Output path for pseudobulk cell proportion CSV")
    parser.add_argument("--out_adata_path", required=True, help="Output path for pseudobulk AnnData (.h5ad)")
    parser.add_argument("--n_cells_per_pseudobulk", type=int, default=800)
    parser.add_argument("--random_state", type=int, default=42)

    # Geneformer-specific arguments
    parser.add_argument("--gene_median_file", default="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/gene_median_dictionary_gc95M.pkl", help="Path to gene median pickle file")
    parser.add_argument("--token_dictionary_file", default="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/token_dictionary_gc95M.pkl", help="Path to token dictionary pickle file")
    parser.add_argument("--gene_mapping_file", default="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/ensembl_mapping_dict_gc95M.pkl", help="Path to Ensembl mapping pickle file")
    parser.add_argument("--geneformer_model_path", default="ctheodoris/Geneformer", help="Path to Geneformer pretrained or fine-tuned model")
    parser.add_argument("--token_output_dir", required=True, help="Directory to save tokenized data")
    parser.add_argument("--token_output_name", required=True, help="Base name for tokenized dataset")
    parser.add_argument("--embedding_output_path", required=True, help="Path to save extracted embeddings CSV, include.csv as in file name")

    args = parser.parse_args()
    main(args)