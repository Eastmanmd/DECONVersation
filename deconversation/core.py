from . import preprocessing,embeddings,deconvolution
import pandas as pd
import scanpy as sc
import warnings
import logging
import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DATASETS_VERBOSITY", "error")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
#os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
import io
from contextlib import redirect_stdout
# ============================================
# Main function for extracting embeddings and deconvolution
# ============================================
def deconverse(
    bulk_df: str,
    model: str,
    mode: str = None,
    adata: str = None,
    sig_df: str = None,
    temp_output_dir: str = "temp",
    cell_type_col: str = "type",
    sample_col: str = "sample",
    solver: str = "nnls",
) -> pd.DataFrame:
    """
    Extracting embeddings for bulk and ref signature data, then run NNLS-based and other methods for deconvolution
Parameters
    ----------
    bulk_df : str
        path to bulk expression matrix (Rows:Genes, Columns:Samples)
    
    adata : str
        path to reference single cell adata object, either adata or precomputed sig_df must be specified
    
    cell_type_col : str
        adata object obs column designating cell type

    sample_col : str
        adata object obs column designating sample or batch

    model : str
        path to zero-shot or fine-tuned model

    mode : str
        scFM model used, support: 
        - "geneformer"
        - "c2s"
        - "cellhermes"
        - "scgpt"
    
    sig_df : str
        path to signature matrix 

    temp_output_dir : str
        path for saving temp files

    solver : str
        solvers currently supported: nnls, ridge, elasticnet, nusvr

    Returns
    -------
    pd.DataFrame
        Estimated cell-type proportions (samples × cell types)
    """
    for noisy in ("transformers", "datasets", "scanpy", "anndata"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    if mode is None:
        mode = embeddings.infer_model(model)
    print("Using model: " + mode)

    # prep ref data
    # make ref signature matrix
    #print("Prepping signature...")
    if sig_df is not None:
        sig_mat = pd.read_csv(sig_df, index_col=0)
    else:
        if adata is None:
            raise ValueError("adata and sig_df cannot both be empty")
        adata = sc.read_h5ad(adata)
        sig_mat = preprocessing.create_signature_matrix(adata = adata,
                                                        sample_col = sample_col,
                                                        cell_type_col = cell_type_col,
                                                        groupby = cell_type_col,
                                                        output_path = None)
        sig_mat.to_csv(temp_output_dir + "/signature.csv")
    if mode == "geneformer" and ("ENS" not in sig_mat.index[0]):
        print("Signature rows are not ENSG ids, converting...")
        sig_mat.index = preprocessing.gene_id_name_map(gene_list=sig_mat.index, mode="to_ensembl" )
    sig_mat = sig_mat.loc[sig_mat.index.dropna()].T
    #sig_mat = sig_mat.T

    # load bulk query data
    #print("Prepping bulk...")
    bulk_df = pd.read_csv(bulk_df, index_col=0)
    if mode == "geneformer" and ("ENS" not in bulk_df.index[0]):
        print("Bulk data rows are not ENSG ids, converting...")
        bulk_df.index = preprocessing.gene_id_name_map(gene_list=bulk_df.index, mode="to_ensembl" )
    bulk_df = bulk_df.loc[bulk_df.index.dropna()].T

    # extract embeddings
    print("Extracting signature embedding...")
    with redirect_stdout(io.StringIO()):
        sig_mat_embed = embeddings.extract_embs(
            bulk_df = sig_mat,
            mode = mode,
            model_path= model,
            temp_output_dir = temp_output_dir + "/sig",
            delete_temp_files = False
        )
    sig_mat_embed.to_csv(temp_output_dir + "/signature_embedding.csv")
    print("Extracting bulk embedding...")
    with redirect_stdout(io.StringIO()):
        bulk_embed = embeddings.extract_embs(
            bulk_df = bulk_df,
            mode = mode,
            model_path= model,
            temp_output_dir = temp_output_dir + "/bulk",
            delete_temp_files = False
        )
    bulk_embed.to_csv(temp_output_dir + "/bulk_embedding.csv")
    
    # solve
    print("Solving deconvolution...")
    cell_prop_pred = deconvolution.run_deconv(bulk_df = bulk_embed.T,
                                signature_df = sig_mat_embed.T, 
                                solver= solver)

    return cell_prop_pred
