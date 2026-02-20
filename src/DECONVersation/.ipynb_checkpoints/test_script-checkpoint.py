#!/gpfs/commons/groups/compbio/projects/rf_condas/cell2sentence/bin/python

import os
from datetime import datetime
import random
from collections import Counter
import numpy as np
from tqdm import tqdm
from transformers import TrainingArguments
import anndata
import scanpy as sc
import cell2sentence as cs
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
#import torch

input_csv = "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvBench/tm_full_signature_matrix_symbol.csv"
c2s_save_dir1="/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/GSE220608/c2s/410m/temp/"
c2s_save_name1 = "temp_embs"
model_path1 = "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvbench/c2s/410mmouse/2026-02-08-08_41_26_finetune_cell_type_prediction/checkpoint-9090"
model_save_dir1 = "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/GSE220608/c2s/410m/mouse/temp"
model_save_name1 = "temp_embs_prediction"

def get_embedding_c2s(
    input_csv: str,
    c2s_save_dir: str,
    c2s_save_name: str,
    model_path: str,
    model_save_dir: str,
    model_save_name: str,
    transpose: bool = False,
    gene_name_rm: str | None = r"\..+",
    use_genes: list[str] | None = None,
    gene_name: list[str] | None = None,
    reorder_obs_name: bool = False,
    n_genes: int = 200,
    log: bool = True,
    log_path: str | None = None,
):
    """
    Generate Cell2Sentence (C2S) embeddings from bulk or pseudobulk expression data.

    Parameters
    ----------
    input_csv : str
        Path to expression matrix (samples x genes). 
    c2s_save_dir : str
        Directory to save CSData object.
    c2s_save_name : str
        Name for saved CSData dataset.
    model_path : str
        Path to pretrained Cell2Sentence model.
    model_save_dir : str
        Directory to save CSModel outputs.
    model_save_name : str
        Name for saved model instance.
    transpose : bool
        Transpose input matrix before processing.
    gene_name_rm : str, optional
        Regex to remove gene suffixes (e.g., Ensembl version numbers).
    use_genes : list of str, optional
        Subset to specific genes.
    gene_name : list of str, optional
        Replace gene names with provided list.
    reorder_obs_name : bool
        If True, rename samples as sample_1, sample_2, ...
    n_genes : int
        Number of genes to use for embedding (default: 200).
    log : bool
        Enable logging and memory tracking.
    log_path : str, optional
        Custom log file path.

    Returns
    -------
    pd.DataFrame
        DataFrame of embeddings indexed by sample name.
    """

    # -----------------------------
    # Logging setup
    # -----------------------------
    if log:
    
        if log_path is None:
            raise ValueError(
                "log=True but no log_path was provided. "
                "Please specify a valid log file path."
            )
    
        # Create parent directory if needed
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w",
            force=True,
        )
    
        logger = logging.getLogger(__name__)
        logger.info("Starting C2S embedding pipeline")
    
        tracemalloc.start()

    # -----------------------------
    # Load data
    # -----------------------------
    data = pd.read_csv(input_csv, index_col=0)

    if transpose:
        data = data.T

    if gene_name_rm is not None:
        data.columns = data.columns.str.replace(gene_name_rm, "", regex=True)

    if use_genes is not None:
        missing = set(use_genes) - set(data.columns)
        if missing:
            raise ValueError(f"Some requested genes not found: {missing}")
        data = data[use_genes]

    data = data.loc[:, ~data.columns.duplicated()]

    # -----------------------------
    # Convert to AnnData
    # -----------------------------
    adata = sc.AnnData(data)

    if gene_name is not None:
        adata.var_names = gene_name

    # Required metadata for C2S
    adata.obs["organism"] = "Homo sapiens"
    adata.obs["cell_type"] = "unknown"
    adata.obs["tissue"] = "unknown"
    adata.obs["sex"] = "unknown"
    adata.obs["batch_condition"] = "unknown"

    label_cols = ["organism", "cell_type", "tissue", "sex", "batch_condition"]

    # -----------------------------
    # Create CSData object
    # -----------------------------
    if log:
        logger.info("Preparing CSData object")

    arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
        adata=adata,
        random_state=42,
        sentence_delimiter=" ",
        label_col_names=label_cols,
    )

    csdata = cs.CSData.csdata_from_arrow(
        arrow_dataset=arrow_ds,
        vocabulary=vocabulary,
        save_dir=c2s_save_dir,
        save_name=c2s_save_name,
        dataset_backend="arrow",
    )

    # -----------------------------
    # Load Model
    # -----------------------------
    csmodel = cs.CSModel(
        model_name_or_path=model_path,
        save_dir=model_save_dir,
        save_name=model_save_name,
    )

    # -----------------------------
    # Extract Embeddings
    # -----------------------------
    if log:
        logger.info("Extracting embeddings")

    embedded_cells = embed_cells(
        csdata=csdata,
        csmodel=csmodel,
        n_genes=n_genes,
    )

    embeddings_df = pd.DataFrame(embedded_cells)

    if reorder_obs_name:
        embeddings_df["name"] = [f"sample_{i}" for i in range(1, df.shape[0] + 1)]
    else:
        embeddings_df["name"] = adata.obs_names.to_list()

    embeddings_df = df.set_index("name")

    if log:
        logger.info("Generating embedding complete")
        snapshot = tracemalloc.take_snapshot()
        logger.info(f"Memory snapshot collected")

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            logger.info(
                f"GPU Memory used: {(total - free) / 1024 / 1024:.2f} MiB"
            )

        logger.info("Pipeline complete")

    return embeddings_df


def main():

    # get embeddings
    c2s_emb = get_embedding_c2s(input_csv = input_csv,
                                c2s_save_dir = c2s_save_dir1,
                                c2s_save_name = c2s_save_name1,
                                model_path = model_path1,
                                model_save_dir = model_save_dir1,
                                model_save_name = model_save_name1,
                                log = False)

    c2s_emb.to_csv("/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/GSE220608/c2s/410m/temp/test_final_embeddings.csv", index=True)


if __name__ == "__main__":
    main()

