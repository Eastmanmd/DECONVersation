#!/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python

"""
Generate Geneformer embeddings from a pseudobulk expression matrix.

The input:
-----------
- A CSV file where rows = samples (pseudobulks), columns = Ensembl gene IDs.

The output:
------------
- A CSV file containing Geneformer embeddings for each pseudobulk sample.

Example usage:
--------------
python pseudobulk_to_geneformer_embeddings.py \
    --input_csv /path/to/pseudobulk.csv \
    --gene_median_file /path/to/gene_median.pkl \
    --token_dictionary_file /path/to/gene_token_dict.pkl \
    --gene_mapping_file /path/to/gene_mapping.pkl \
    --geneformer_model_path /path/to/geneformer_model_directory \
    --token_output_dir /path/to/output_tokens \
    --token_output_name pseudobulk_tokens \
    --embedding_output_path /path/to/output_embeddings.csv
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from geneformer.tokenizer import TranscriptomeTokenizer
from geneformer import perturber_utils as pu
from geneformer import EmbExtractor, TranscriptomeTokenizer, get_embs


def main():
    parser = argparse.ArgumentParser(description="Generate Geneformer embeddings from a pseudobulk CSV matrix.")
    parser.add_argument("--input_csv", required=True, help="Path to input pseudobulk CSV (samples x Ensembl IDs).")
    parser.add_argument("--gene_median_file", default="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/gene_median_dictionary_gc95M.pkl", help="Path to gene median expression file (.pkl).")
    parser.add_argument("--token_dictionary_file", default="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/token_dictionary_gc95M.pkl", help="Path to Geneformer token dictionary (.pkl).")
    parser.add_argument("--gene_mapping_file", default="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/ensembl_mapping_dict_gc95M.pkl", help="Path to Geneformer gene mapping file (.pkl).")
    parser.add_argument("--geneformer_model_path", default="ctheodoris/Geneformer", help="Path to pretrained Geneformer model directory.")
    parser.add_argument("--token_output_dir", required=True, help="Directory to save tokenized outputs.")
    parser.add_argument("--token_output_name", required=True, help="Base name for tokenized dataset files.")
    parser.add_argument("--embedding_output_path", required=True, help="Path to save the final embeddings CSV.")
    args = parser.parse_args()

    # ----------------------------- #
    # Load and prepare pseudobulk data
    # ----------------------------- #
    print("Loading pseudobulk CSV...")
    pseudobulk_df = pd.read_csv(args.input_csv, index_col=0)

    # Convert to AnnData
    pb_adata = sc.AnnData(pseudobulk_df)
    pb_adata.obs["cell_type"] = "unknown"
    pb_adata.obs["n_counts"] = np.sum(pb_adata.X, axis=1).tolist()
    pb_adata.var["ensembl_id"] = pb_adata.var_names
    pb_adata.X = sp.csc_matrix(pb_adata.X)

    # Save to temporary .h5ad for tokenization
    out_adata_path = os.path.join(args.token_output_dir, f"{args.token_output_name}.h5ad")
    os.makedirs(args.token_output_dir, exist_ok=True)
    pb_adata.write_h5ad(out_adata_path)

    print(f"Pseudobulk AnnData saved to: {out_adata_path}")

    # ----------------------------- #
    # Tokenization step
    # ----------------------------- #
    print("\nStarting Geneformer tokenization...")

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
        os.path.dirname(out_adata_path),
        args.token_output_dir,
        args.token_output_name,
        file_format="h5ad"
    )

    # ----------------------------- #
    # Load Geneformer model
    # ----------------------------- #
    print("\nLoading Geneformer model...")
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

    # ----------------------------- #
    # Load tokenized dataset
    # ----------------------------- #
    print("\nLoading tokenized dataset...")
    filtered_input_data = pu.load_and_filter(
        filter_data=None,
        nproc=1,
        input_data_file=f"{args.token_output_dir}/{args.token_output_name}.dataset/"
    )

    # ----------------------------- #
    # Extract embeddings
    # ----------------------------- #
    print("\nExtracting Geneformer embeddings...")
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

    # Convert embeddings to DataFrame
    tempout = pd.DataFrame(state_embs_dict.cpu().numpy())
    df_meta = filtered_input_data.to_pandas()
    #tempout.index = df_meta["cell_type"] + "_" + df_meta.index.astype(str)
    #tempout = tempout.reset_index(drop=True)

    # Save embeddings
    tempout.index = pseudobulk_df.index
    tempout.to_csv(args.embedding_output_path, index=True)
    print(f"\n✅ Geneformer embeddings saved to: {args.embedding_output_path}")


if __name__ == "__main__":
    main()