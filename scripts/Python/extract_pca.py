"""
Compute principal components of the pseudobulk data.

This scripts:
1. Reads in pseudobulk for training, test and signature matrices 
2. Transforms pseudobulk into principal components (PCs)
3. Saves PCA data to specified paths

Example usage:
------------------

python pca_transform.py \
    --train_pseudobulk /path/to/train/pseudobulk.csv \
    --test_pseudobulk /path/to/test/pseudobulk.csv \
    --signature_matrix /path/to/signature_matrix.csv \
    --outdir /path/to/output_pca \
    --n_components 100
    
"""

import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
import os


def run_pca(train_path, test_path, sig_path, outdir, outdir_test, n_components):

    # -------------------------
    # Set up logger
    # -------------------------
    logging.basicConfig(format='%(asctime)s %(message)s', 
                        filename=args.log_file,
                        filemode='w', force = True)

    # -------------------------
    # Load data
    # -------------------------
    logging.warning('Load data')
    pseudobulk_train = pd.read_csv(train_path, index_col=0)
    pseudobulk_test = pd.read_csv(test_path, index_col=0)
    sig_matrix = pd.read_csv(sig_path, index_col=0)

    # -------------------------
    # Scaling (fit only on train)
    # -------------------------
    logging.warning('Scale and transform data')
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(pseudobulk_train)
    test_scaled = scaler.transform(pseudobulk_test)
    sig_scaled = scaler.transform(sig_matrix)

    # -------------------------
    # PCA (fit only on train)
    # -------------------------
    logging.warning('Fit PCA')
    pca = PCA(n_components=n_components, random_state=42)
    pca_train = pca.fit_transform(train_scaled)
    pca_test = pca.transform(test_scaled)
    pca_sig = pca.transform(sig_scaled)

    # Convert to DataFrames
    pca_train = pd.DataFrame(pca_train, index=pseudobulk_train.index)
    pca_test = pd.DataFrame(pca_test, index=pseudobulk_test.index)
    pca_sig = pd.DataFrame(pca_sig, index=sig_matrix.index)

    # -------------------------
    # Save outputs
    # -------------------------
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_test, exist_ok=True)

    logging.warning('Saving data')
    pca_train.to_csv(os.path.join(outdir, "pca_train.csv"))
    pca_test.to_csv(os.path.join(outdir_test, "pca_test.csv"))
    pca_sig.to_csv(os.path.join(outdir, "pca_signature_matrix.csv"))

    print(f"Saved PCA files to: {outdir}")
    print(f"Saved Test PCA files to: {outdir_test}")


# -----------------------------------------------------
# Argument parser
# -----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA transform pseudobulk and signature matrices")

    parser.add_argument("--train_pseudobulk", required=True,
                        help="Path to training pseudobulk CSV")

    parser.add_argument("--test_pseudobulk", required=True,
                        help="Path to test pseudobulk CSV")

    parser.add_argument("--signature_matrix", required=True,
                        help="Path to signature matrix CSV")

    parser.add_argument("--outdir", required=True,
                        help="Directory to save PCA-transformed outputs")

    parser.add_argument("--outdir_test", required=True,
                    help="Directory to save test PCA-transformed outputs")

    parser.add_argument("--n_components", type=int, default=100,
                        help="Number of PCA components (default: 100)")

    args = parser.parse_args()

    run_pca(
        train_path=args.train_pseudobulk,
        test_path=args.test_pseudobulk,
        sig_path=args.signature_matrix,
        outdir=args.outdir,
        outdir_test = args.outdir_test,
        n_components=args.n_components
    )


    