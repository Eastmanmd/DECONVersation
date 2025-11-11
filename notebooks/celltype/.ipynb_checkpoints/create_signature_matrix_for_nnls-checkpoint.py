"""
Generate a signature matrix from single-cell RNA-seq data.

This script:
------------
1. Loads an AnnData (.h5ad) file
2. Subsets to specified samples (using a user-defined column)
3. Optionally filters to specific cell types (using a user-defined column)
4. Groups cells by a specified column (e.g., cell type)
5. Computes the average expression per gene per group
6. Saves the resulting signature matrix as a CSV

Example usage:
--------------
python create_signature_matrix.py \
    --adata_path /path/to/input_data.h5ad \
    --sample_col id \
    --sample_ids B2-CZI08N B3-CZI11N B3-CZI05N \
    --celltype_col newtype \
    --celltypes Endothelial EpithelialGL_Hormonal StromalEMT \
    --groupby newtype \
    --output_path /path/to/output/signature_matrix.csv
"""

import argparse
import pandas as pd
import scanpy as sc


def create_signature_matrix(adata, groupby):
    """
    Compute average expression per gene per cell group.
    """
    # Convert counts to dense DataFrame if sparse
    if hasattr(adata.X, "toarray"):
        expr = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
    else:
        expr = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

    # Add grouping column
    expr[groupby] = adata.obs[groupby].values

    # Average expression per group (genes × groups)
    signature = expr.groupby(groupby).mean().T
    return signature





def main():
    parser = argparse.ArgumentParser(description="Create a gene expression signature matrix from single-cell data.")
    parser.add_argument("--adata_path", required=True, help="Path to the input .h5ad file.")
    parser.add_argument("--sample_col", required=True, help="Column in adata.obs that contains sample identifiers.")
    parser.add_argument("--sample_ids", nargs="+", required=True, help="List of sample IDs to subset.")
    parser.add_argument("--celltype_col", required=True, help="Column in adata.obs that contains cell type labels.")
    parser.add_argument("--celltypes", nargs="+", required=True, help="List of cell types to include in the analysis.")
    parser.add_argument("--groupby", required=True, help="Column in adata.obs to group by (e.g., cell type).")
    parser.add_argument("--output_path", required=True, help="Path to save the resulting signature matrix (CSV).")
    args = parser.parse_args()

    # Read AnnData
    print(f"Reading AnnData object from: {args.adata_path}")
    adata = sc.read_h5ad(args.adata_path)

    # Add ensembl ID information
    gene_ids = pd.read_csv("/gpfs/commons/groups/compbio/projects/CZI_endom/RNA_temp/gene_names_gene_ids_czi_endo.csv", index_col= False)
    
    gene_id_dict = pd.DataFrame({
        'gene_symbol': gene_ids["gene_name"],
        'ensembl_id': gene_ids["gene_id"]})
    
    gene_id_dict = gene_id_dict.set_index('gene_symbol')['ensembl_id'].to_dict()
    
    ensembl_ids = [gene_id_dict.get(gene, 'NA') for gene in adata.var_names]
    
    # Add ensembl IDs to anndata object
    adata.var["gene_names"] = adata.var_names
    adata.var["ensembl_id"] = ensembl_ids
    adata.var_names = ensembl_ids
    
    # Remove duplicated IDs
    adata = adata[:, ~adata.var["ensembl_id"].duplicated()]

    # Validate columns
    for col in [args.sample_col, args.celltype_col, args.groupby]:
        if col not in adata.obs.columns:
            raise ValueError(f"The specified column '{col}' is not found in adata.obs.")

    # Subset to specified samples
    print(f"Subsetting to {len(args.sample_ids)} sample(s) using column '{args.sample_col}'...")
    adata = adata[adata.obs[args.sample_col].isin(args.sample_ids)].copy()
    if adata.n_obs == 0:
        raise ValueError("No cells match the provided sample IDs after subsetting.")

    # Subset to specified cell types
    print(f"Filtering to {len(args.celltypes)} cell type(s) from column '{args.celltype_col}'...")
    adata = adata[adata.obs[args.celltype_col].isin(args.celltypes)].copy()
    if adata.n_obs == 0:
        raise ValueError("No cells match the provided cell types after subsetting.")

    # Compute signature matrix
    print(f"Creating signature matrix grouped by '{args.groupby}'...")
    signature = create_signature_matrix(adata, groupby=args.groupby)
    signature = signature.T # colnmaes should be ensemblIDs, rownames samples

    # Save results
    signature.to_csv(args.output_path)
    print(f"✅ Signature matrix saved to: {args.output_path}")


if __name__ == "__main__":
    main()