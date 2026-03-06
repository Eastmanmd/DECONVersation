# ===============================
# Standard Libraries
# ===============================
import os
import shutil
import pickle
import logging
import tracemalloc

# ===============================
# For Data/Model Manipulation
# ===============================
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

# ===============================
# Geneformer
# ===============================
from geneformer import TranscriptomeTokenizer
from geneformer import perturber_utils as pu
from geneformer.emb_extractor import get_embs

# ===============================
# Cell2Sentence
# ===============================
#import cell2sentence as cs
#from cell2sentence.tasks import embed_cells


# -----------------------------
# Extract geneformer embeddings 
# -----------------------------
def extract_embs(
    bulk_df,
    mode,
    temp_output_dir, 
    model_path,
    delete_temp_files = False
):
    if mode == "geneformer":
        emb = extract_geneformer_embs(
            bulk_df = bulk_df,
            token_output_dir = temp_output_dir,
            token_output_name = "gf_tokens",
            geneformer_model_path = model_path,
            delete_temp_files = delete_temp_files
        )

    elif mode == "c2s":
        emb = get_embedding_c2s(
            bulk_df = bulk_df,
            c2s_save_dir = temp_output_dir,
            c2s_save_name = "c2s_object",
            model_path1 = model_path,
            model_save_dir = temp_output_dir,
            model_save_name = "c2s_model")

    else:
        raise ValueError("mode must be either 'geneformer' or 'c2s' ")
        
    return emb




# -----------------------------
# Extract geneformer embeddings 
# -----------------------------
def extract_geneformer_embs(
    bulk_df,
    token_output_dir,
    token_output_name,
    delete_temp_files,
    geneformer_model_path,
    gene_median_file="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/gene_median_dictionary_gc95M.pkl",
    token_dictionary_file="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/token_dictionary_gc95M.pkl",
    gene_mapping_file="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/ensembl_mapping_dict_gc95M.pkl" 
    
):

    """
    Generate Geneformer embeddings from a bulk CSV matrix.

    Parameters
    ----------
    input_csv : str
        Path to input pseudobulk CSV (samples x Ensembl IDs).
    token_output_dir : str
        Directory to save tokenized outputs.  
    token_output_name : str
        Base name for tokenized dataset files.
    gene_median_file : str
        Path to gene median expression file (.pkl).
    token_dictionary_file : str
        Path to Geneformer token dictionary (.pkl).
    gene_mapping_file : str
        Path to Geneformer gene mapping file (.pkl).
    geneformer_model_path : str
        Path to pretrained Geneformer model directory.
    """

    #print("Loading pseudobulk CSV...")  # Add code to ensure that the colnames are ensemble IDs
    #bulk_df = pd.read_csv(input_csv, index_col=0)

    # Ensure column names are Ensembl IDs
    if not all(col.startswith("ENSG") for col in bulk_df.columns):
        raise ValueError(
            "Input CSV columns must be Ensembl gene IDs (e.g., ENSG00000123456). "
            "Detected non-Ensembl column names."
        )

    # -----------------------------
    # Convert to AnnData
    # -----------------------------
    pb_adata = sc.AnnData(bulk_df)
    pb_adata.obs["cell_type"] = "unknown" 
    pb_adata.obs["n_counts"] = np.sum(pb_adata.X, axis=1).tolist()
    pb_adata.var["ensembl_id"] = pb_adata.var_names
    pb_adata.X = sp.csc_matrix(pb_adata.X)

    # Save temporary .h5ad
    out_adata_path = os.path.join(token_output_dir, f"{token_output_name}.h5ad")
    os.makedirs(token_output_dir, exist_ok=True)
    pb_adata.write_h5ad(out_adata_path)

    print(f"Pseudobulk AnnData saved to: {out_adata_path}")

    # -----------------------------
    # Tokenization 
    # -----------------------------
    print("Starting Geneformer tokenization...")

    tk = TranscriptomeTokenizer(
        {"cell_type": "cell_type"},
        model_input_size = 4096,
        special_token = True,
        chunk_size = 512,
        gene_median_file = gene_median_file,
        token_dictionary_file = token_dictionary_file,
        gene_mapping_file = gene_mapping_file
    )

    tk.tokenize_data(
        os.path.dirname(out_adata_path),
        token_output_dir,
        token_output_name,
        file_format="h5ad"
    )

    # -----------------------------
    # Load  model
    # -----------------------------
    print("Loading Geneformer model...")
    model = pu.load_model(
        model_type = "Pretrained",
        num_classes = 0,
        model_directory = geneformer_model_path,
        mode="eval"
    )

    with open(token_dictionary_file, "rb") as f:
        gene_token_dict = pickle.load(f)
        
    token_gene_dict = {v: k for k, v in gene_token_dict.items()}
    pad_token_id = gene_token_dict.get("<pad>")

    # -----------------------------
    # Load tokenized dataset
    # -----------------------------
    print("Loading tokenized dataset...")
    
    filtered_input_data = pu.load_and_filter(
        filter_data=None,
        nproc=1,
        input_data_file=f"{token_output_dir}/{token_output_name}.dataset/"
    )

    # -----------------------------
    # Extract embeddings
    # -----------------------------
    
    print("Extracting Geneformer embeddings...")
    state_embs_dict = get_embs(
        model,
        filtered_input_data,
        emb_mode="cell",
        layer_to_quant=18,
        pad_token_id=pad_token_id,
        token_gene_dict=token_gene_dict,
        special_token=True,
        forward_batch_size=50
    )

    # Return embeddings as dataframe
    embeddings_df = pd.DataFrame(state_embs_dict.cpu().numpy())
    embeddings_df.index = bulk_df.index

    # -----------------------------
    # Delete temp files
    # -----------------------------
    if delete_temp_files:
        for filename in os.listdir(token_output_dir):
            file_path = os.path.join(token_output_dir, filename)
            
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    return embeddings_df


# --------------------------------
# Extract cell2sentence embeddings 
# --------------------------------
def get_embedding_c2s(
    bulk_df: str,
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
    if transpose:
        bulk_df = bulk_df.T

    if gene_name_rm is not None:
        bulk_df.columns = data.columns.str.replace(gene_name_rm, "", regex=True)

    if use_genes is not None:
        missing = set(use_genes) - set(data.columns)
        if missing:
            raise ValueError(f"Some requested genes not found: {missing}")
        bulk_df = bulk_df[use_genes]

    bulk_df = bulk_df.loc[:, ~bulk_df.columns.duplicated()]

    # -----------------------------
    # Convert to AnnData
    # -----------------------------
    adata = sc.AnnData(bulk_df)

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
        embeddings_df["name"] = [f"sample_{i}" for i in range(1, embeddings_df.shape[0] + 1)]
    else:
        embeddings_df["name"] = adata.obs_names.to_list()

    embeddings_df = embeddings_df.set_index("name")

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