#!/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python

"""
Run NNLS deconvolution on pseudobulk data using a given signature matrix.

Usage:
    python run_nnls.py \
        --pseudobulk pseudobulk_embeddings.csv \
        --signature signature_matrix.csv \
        --output cell_proportions.csv

Inputs:
    pseudobulk : CSV file where rows = genes (Ensembl IDs), columns = samples
    signature  : CSV file where rows = genes (same IDs), columns = cell types
Output:
    A CSV file containing estimated cell-type proportions for each sample
"""

import pandas as pd
import numpy as np
from scipy.optimize import nnls
import argparse
import os

# ----------------------------- #
# Function to estimate proportions
# ----------------------------- #
def estimate_proportions(bulk_df, signature_matrix):
    """
    Estimate cell type proportions in bulk samples via NNLS.

    Parameters
    ----------
    bulk_df : pd.DataFrame
        Bulk expression (genes x samples).
    signature_matrix : pd.DataFrame
        Signature matrix (genes x cell types).

    Returns
    -------
    pd.DataFrame
        Proportions (samples x cell types).
    """
    # Keep only common genes
    common_genes = bulk_df.index.intersection(signature_matrix.index)
    if len(common_genes) == 0:
        raise ValueError("No common genes found between pseudobulk and signature matrix.")

    bulk = bulk_df.loc[common_genes]
    sig = signature_matrix.loc[common_genes]

    props = []
    for sample in bulk.columns:
        y = bulk[sample].values
        x = sig.values
        coeffs, _ = nnls(x, y)
        if coeffs.sum() > 0:
            coeffs = coeffs / coeffs.sum()
        props.append(coeffs)

    props_df = pd.DataFrame(props, index=bulk.columns, columns=sig.columns)
    return props_df


# ----------------------------- #
# Main script
# ----------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Run NNLS deconvolution.")
    parser.add_argument("--pseudobulk", required=True, help="Path to pseudobulk embeddings CSV file")
    parser.add_argument("--signature", required=True, help="Path to signature matrix CSV file")
    parser.add_argument("--output", required=True, help="Path to save estimated proportions CSV file")

    args = parser.parse_args()

    # Read input data
    print("Reading pseudobulk and signature matrix...")
    pseudobulk_df = pd.read_csv(args.pseudobulk, index_col=0)
    signature_df = pd.read_csv(args.signature, index_col=0)

    # Transpose to match nnls function
    pseudobulk_df = pseudobulk_df.T
    signature_df = signature_df.T

    # Add colnames to map embedding dimensions
    pseudobulk_df.index = ["embed_" + str(i) for i in pseudobulk_df.index]
    signature_df.index = ["embed_" + str(i) for i in signature_df.index]

    # Run NNLS deconvolution
    print("Running NNLS deconvolution...")
    nnls_props = estimate_proportions(bulk_df=pseudobulk_df, signature_matrix=signature_df)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    nnls_props.to_csv(args.output)
    print(f"\n✅ Cell-type proportions saved to: {args.output}")


if __name__ == "__main__":
    main()
