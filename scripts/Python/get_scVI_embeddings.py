"""
Generate scVI embeddings from a pseudobulk expression matrix.

The input:
-----------
- A CSV file where rows = samples (pseudobulks), columns = Ensembl gene IDs.

The output:
------------
- A CSV file containing scVI embeddings for each pseudobulk sample.

Example usage:
--------------
python pseudobulk_to_scvi_embeddings.py \
    --input_csv /path/to/pseudobulk.csv \
    --scvi_model_path /gpfs/commons/groups/compbio/projects/ao_projects/scvi-human-2024-07-01 \
    --embedding_output_path /path/to/output_embeddings.csv
"""

import os
import argparse
import pandas as pd
import anndata
import scvi


def main():
    parser = argparse.ArgumentParser(description="Generate scVI embeddings from a pseudobulk CSV matrix.")
    parser.add_argument("--input_csv", required=True, help="Path to input pseudobulk CSV (samples x Ensembl IDs).")
    parser.add_argument("--scvi_model_path", default="/gpfs/commons/groups/compbio/projects/ao_projects/scvi-human-2024-07-01",
                        help="Path to trained scVI model directory.")
    parser.add_argument("--embedding_output_path", required=True, help="Path to save the final embeddings CSV.")
    args = parser.parse_args()

    # ----------------------------- #
    # Load pseudobulk data
    # ----------------------------- #
    print("Loading pseudobulk CSV...")
    pseudobulk_df = pd.read_csv(args.input_csv, index_col=0)

    # Convert to AnnData
    print("Converting pseudobulk matrix to AnnData...")
    adata_bulk = anndata.AnnData(pseudobulk_df)
    adata_bulk.obs_names = pseudobulk_df.index
    adata_bulk.var_names = pseudobulk_df.columns
    adata_bulk.obs["batch"] = "pseudobulk"
    adata_bulk.obs["id"] = pseudobulk_df.index

    # ----------------------------- #
    # Prepare and load scVI model
    # ----------------------------- #
    print("\nPreparing query with pretrained scVI model...")
    scvi.model.SCVI.prepare_query_anndata(adata_bulk, args.scvi_model_path)

    print("Loading scVI pretrained model...")
    vae_q = scvi.model.SCVI.load_query_data(adata_bulk, args.scvi_model_path)
    vae_q.is_trained = True

    # ----------------------------- #
    # Get latent representations
    # ----------------------------- #
    print("\nExtracting scVI latent embeddings...")
    latent = vae_q.get_latent_representation()

    latent_df = pd.DataFrame(latent, index=adata_bulk.obs_names)
    #latent_df.index.name = "sample_id"

    # ----------------------------- #
    # Save embeddings
    # ----------------------------- #
    latent_df.to_csv(args.embedding_output_path)
    print(f"\n✅ scVI embeddings saved to: {args.embedding_output_path}")


if __name__ == "__main__":
    main()