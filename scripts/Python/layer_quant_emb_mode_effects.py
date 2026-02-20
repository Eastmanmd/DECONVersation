#!/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python

import os
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from geneformer.tokenizer import TranscriptomeTokenizer
from geneformer import perturber_utils as pu
from geneformer import get_embs


INPUT_FILES = [
    "/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/pseudo_bulk/pseudo_bulk_counts_gf.csv"]

# For sig mat 
# File:                  /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/pseudo_bulk/pb_sig_matrix.csv"
# TOKEN_OUTPUT_DIR       /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_hoek/train/tokenize_finetuned/sig_mat/
# EMBEDDING_OUTPUT_DIR    /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/pseudo_bulk/emb_mode_files_finetuned/sig_mat/

# For zeroshot
# replace tokenize_finetuned: tokenize
# emb_mode_files_finetuned:     emb_mode_files

TOKEN_OUTPUT_DIR = "/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_hoek/train/tokenize_finetuned/pseudobulk/"
TOKEN_OUTPUT_NAME = "temp_tokens"
EMBEDDING_OUTPUT_DIR = "/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/pseudo_bulk/emb_mode_files_finetuned/pseudobulk/"

GENE_MEDIAN_FILE = "/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/gene_median_dictionary_gc95M.pkl"
TOKEN_DICTIONARY_FILE = "/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/token_dictionary_gc95M.pkl"
GENE_MAPPING_FILE = "/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/ensembl_mapping_dict_gc95M.pkl"
GENEFORMER_MODEL_PATH = "/gpfs/commons/groups/compbio/projects/CZI_endom/geneformer/output/czi_perturb_noStemPro/rand_undersamp_v2_316M_1500/251103_geneformer_cellClassifier_cell_annot_subset/ksplit1/"

#/gpfs/commons/groups/compbio/projects/CZI_endom/geneformer/output/czi_perturb_noStemPro/rand_undersamp_v2_316M_1500/251103_geneformer_cellClassifier_cell_annot_subset/ksplit1/
#ctheodoris/Geneformer

EMB_MODES = ["cls", "cell"]
LAYERS = list(range(1, 19))  #change to include zero

# Tokenize pseudobulk
def tokenize_pseudobulk(pseudobulk_df, token_output_dir, token_output_name):

    pb_adata = sc.AnnData(pseudobulk_df)
    pb_adata.obs["cell_type"] = "unknown"
    pb_adata.obs["n_counts"] = np.sum(pb_adata.X, axis=1).tolist()
    pb_adata.var["ensembl_id"] = pb_adata.var_names
    pb_adata.X = sp.csc_matrix(pb_adata.X)

    os.makedirs(token_output_dir, exist_ok=True)
    out_adata_path = os.path.join(token_output_dir, f"{token_output_name}.h5ad")
    pb_adata.write_h5ad(out_adata_path)

    tk = TranscriptomeTokenizer(
        {"cell_type": "cell_type"},
        model_input_size=4096,
        special_token=True,
        chunk_size=512,
        gene_median_file=GENE_MEDIAN_FILE,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
        gene_mapping_file=GENE_MAPPING_FILE
    )

    tk.tokenize_data(
        os.path.dirname(out_adata_path),
        token_output_dir,
        token_output_name,
        file_format="h5ad"
    )

    return out_adata_path


# Main function
def main():

    os.makedirs(EMBEDDING_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TOKEN_OUTPUT_DIR, exist_ok=True)

    print("Loading Geneformer model...")
    model = pu.load_model(
        model_type="Pretrained",
        num_classes=0,
        model_directory=GENEFORMER_MODEL_PATH,
        mode="eval"
    )

    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        gene_token_dict = pickle.load(f)

    token_gene_dict = {v: k for k, v in gene_token_dict.items()}
    pad_token_id = gene_token_dict.get("<pad>")

    for input_csv in INPUT_FILES:

        base = os.path.splitext(os.path.basename(input_csv))[0]
        print(f"Processing: {base}")

        pseudobulk_df = pd.read_csv(input_csv, index_col=0)

        token_name = f"{base}_tokens"
        tokenize_pseudobulk(pseudobulk_df, TOKEN_OUTPUT_DIR, token_name)

        print("Loading tokenized dataset...")
        filtered_input_data = pu.load_and_filter(
            filter_data=None,
            nproc=1,
            input_data_file=f"{TOKEN_OUTPUT_DIR}/{token_name}.dataset/"
        )

        for emb_mode in EMB_MODES:
            for layer in LAYERS:

                print(f"Embedding: {base} | mode={emb_mode} | layer={layer}")

                state_embs_dict = get_embs(
                    model,
                    filtered_input_data,
                    emb_mode=emb_mode,
                    layer_to_quant=layer,
                    pad_token_id=pad_token_id,
                    token_gene_dict=token_gene_dict,
                    special_token=True,
                    forward_batch_size=50
                )

                emb_df = pd.DataFrame(state_embs_dict.cpu().numpy())
                emb_df.index = pseudobulk_df.index

                out_path = os.path.join(
                    EMBEDDING_OUTPUT_DIR,
                    f"{base}_emb_mode_{emb_mode}_layer_{layer}.csv"
                )

                emb_df.to_csv(out_path)
                print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

    