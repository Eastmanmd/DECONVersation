#!/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python

import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import nnls


SIG_MATRIX_DIR = "/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/pseudo_bulk/emb_mode_files/sig_mat/"
PSEUDOBULK_DIR = "/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/pseudo_bulk/emb_mode_files/pseudobulk/"
OUTPUT_DIR = "/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/pseudo_bulk/emb_mode_files/results/"

EMB_MODES = ["cell", "cls"]
LAYERS = list(range(1, 19)) # Also adjist to include 0


def run_nnls(pseudobulk_df, sig_matrix_df):
    """
    Runs NNLS for each pseudobulk sample.
    Returns dataframe of estimated proportions.
    """

    # Ensure features align
    #print(sig_matrix_df.index)
    #print(pseudobulk_df.index)
    
    common_features = pseudobulk_df.index.intersection(sig_matrix_df.index)

    if len(common_features) == 0:
        raise ValueError("No overlapping features between pseudobulk and signature matrix.")

    bulk = pseudobulk_df.loc[common_features]
    sig = sig_matrix_df.loc[common_features]

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


def main():

    print("starting ...")
    
    # Make directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for emb_mode in EMB_MODES:
        for layer in LAYERS:

            print(f"\nProcessing emb_mode={emb_mode}, layer={layer}")

            # Expected file names
            pb_file = os.path.join(
                PSEUDOBULK_DIR,
                f"pseudo_bulk_counts_gf_emb_mode_{emb_mode}_layer_{layer}.csv"
            )

            sig_file = os.path.join(
                SIG_MATRIX_DIR,
                f"pb_sig_matrix_emb_mode_{emb_mode}_layer_{layer}.csv"
            )

            if not os.path.exists(pb_file):
                print(f"Skipping missing pseudobulk: {pb_file}")
                continue

            if not os.path.exists(sig_file):
                print(f"Skipping missing signature matrix: {sig_file}")
                continue

            print("Loading files...")
            pseudobulk_df = pd.read_csv(pb_file, index_col=0)
            sig_matrix_df = pd.read_csv(sig_file, index_col=0)

            # Transpose to match nnls function
            pseudobulk_df = pseudobulk_df.T
            sig_matrix_df = sig_matrix_df.T

            # Add colnames to map embedding dimensions
            pseudobulk_df.index = ["embed_" + str(i) for i in pseudobulk_df.index]
            sig_matrix_df.index = ["embed_" + str(i) for i in sig_matrix_df.index]

            # Run NNLS
            print("Running NNLS...")
            nnls_results = run_nnls(pseudobulk_df, sig_matrix_df)

            # Save output
            out_file = os.path.join(
                OUTPUT_DIR,
                f"nnls_emb_mode_{emb_mode}_layer_{layer}.csv"
            )

            nnls_results.to_csv(out_file)
            print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()