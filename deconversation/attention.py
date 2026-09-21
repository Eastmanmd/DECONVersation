import gc
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import scanpy as sc
import anndata as ad
from datasets import load_from_disk
from transformers import AutoModel


# ===============================
# Geneformer
# ===============================
try:
    from geneformer import TranscriptomeTokenizer
    print("geneformer successfully imported.")
    
except ImportError as e:
    print("geneformer is not installed. Skipping related functions.")



# ---------------------------------
# Extract geneformer attention 
# ---------------------------------
def get_attention_by_gene_gf(
    data, #anndata or dataframe
    model_path,
    output_dir,
    cell_types, # Accepts a single string or a list of cell types
    output_name = "gf_attention",
    cell_type_col = "cell_type",
    model_version = "V2",
    num_of_cells = 50,
    random_state = 42,
    min_cells_per_cell_type = None,
):
    """
    Extract per-gene attention scores from Geneformer for one or more cell types.

    Runs cells through a pretrained or fine-tuned Geneformer model and extracts the
    CLS token's attention over gene tokens from the final layer, then maps token IDs
    back to gene names and averages across cells. The result ranks genes by how much
    the model attends to them when representing each cell type.
    
    Parameters
    ----------
    data : anndata.AnnData or pandas.DataFrame
        Expression data. A DataFrame (e.g. a signature matrix) is converted to
        AnnData internally. Gene identifiers must be Ensembl IDs, as required by
        Geneformer.
    model_path : str or pathlib.Path
        Path to the Geneformer model to use. A local directory (such as a model
        fine-tuned with :func:`train_geneformer_cell_classifier`) or a Hugging Face
        repository ID.
    output_dir : str or pathlib.Path
        Directory for intermediate files. Receives ``{output_name}_data.h5ad`` and a
        ``tokenized.dataset/`` directory.
    cell_types : str or list of str
        Cell type(s) to extract attention for. A single string is accepted. Each
        type is filtered and processed separately.
    output_name : str, default "gf_attention"
        Filename prefix for the intermediate ``.h5ad`` written to ``output_dir``.
    cell_type_col : str, default "cell_type"
        Column in ``data.obs`` holding the cell type labels matched against
        ``cell_types``.
    model_version : str, default "V2"
        Geneformer model version, passed to ``TranscriptomeTokenizer``. Must match
        the model at ``model_path``.
    num_of_cells : int, default 50
        Maximum cells to process per cell type. The effective count is
        ``min(num_of_cells, n_available)``, so fewer are used when a type is rare.
    random_state : int, default 42
        Seed for subsampling, for reproducible cell selection.
    min_cells_per_cell_type : int, optional
        If given (AnnData input only), each cell type group is subsampled to at most
        this many cells before tokenization. Use to cap cost on large datasets.

    Returns
    -------
    dict
        Nested dictionary mapping each cell type to its per-gene mean attention
        scores::

            {cell_type: {gene_name: mean_attention_score, ...}, ...}

        Scores are averaged over the cells processed for that type. Cell types with
        no matching cells are absent from the result.
    """

    # Ensure cell_types is a list even if a single string is passed
    if isinstance(cell_types, str):
        cell_types = [cell_types]

    # Subset adata to min cells per type
    if isinstance(data, ad.AnnData) and min_cells_per_cell_type is not None:
        subsampled_index = (
        data.obs.groupby(cell_type_col, group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min(len(group), min_cells_per_cell_type), random_state=random_state))
        .index)

        # subsetted data 
        data = data[subsampled_index].copy()
        print("new subsetted data shape")
        print(data.shape)

    elif isinstance(data, pd.DataFrame):
        
        # convert to anndata
        data_df = data.copy()
        data = sc.AnnData(data)
        data.obs["cell_type"] = data_df.index
        data.obs["n_counts"] = np.sum(data.X, axis=1).tolist()
        data.var["ensembl_id"] = data.var_names
        data.X = sp.csc_matrix(data.X)


    # Save data in output dir
    os.makedirs(output_dir, exist_ok=True)    
    data.write_h5ad(f"{output_dir}/{output_name}_data.h5ad")

    #------------ Tokenize data ------------------------------
    # Load tokenizer
    print("starting tokenization...")
    tk = TranscriptomeTokenizer(
        {"cell_type": "cell_type"},
        model_input_size=4096,
        special_token=True,
        chunk_size=512,
        model_version = model_version,
    )

    # tokenize data 
    tk.tokenize_data(
        os.path.dirname(output_dir),
        output_dir,
        "tokenized",
        file_format="h5ad",
    )

    print("tokenization done...")

    # Path to tokenized data 
    dataset_path = os.path.join(output_dir, "tokenized.dataset/")
    print(dataset_path)


    #-------------- Extract embeddings ----------------------------
    # select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load geneformer model (zeroshot or finetuned)
    model = AutoModel.from_pretrained(
        model_path,
        output_attentions=True,
    )
    model.eval()
    model.to(device)

    # Get ID to gene dictionary
    gene_token_dict = tk.gene_token_dict 
    id_to_gene = {token_id: gene for gene, token_id in gene_token_dict.items()}

    # Load full dataset
    dataset = load_from_disk(dataset_path)

    results_by_cell_type = {}

    # Iterate through all requested cell types
    for cell_type in cell_types:
        filtered_dataset = dataset.filter(
            lambda sample: sample["cell_type"] == cell_type
        )

        # Number of cells to process (limited to sample size of bulk)
        num_cells_to_process = min(num_of_cells, len(filtered_dataset))
        subset = filtered_dataset.select(range(num_cells_to_process))

        print(
            f"Processing {len(subset)} cells "
            f"for cell type '{cell_type}'..."
        )

        all_attn = {}

        with torch.inference_mode():
            for i, sample in enumerate(subset):
                input_ids = sample["input_ids"]
                seq_len = len(input_ids)

                if seq_len <= 1:
                    continue

                input_tensor = None
                outputs = None
                avg_heads = None

                try:
                    input_tensor = torch.tensor(
                        [input_ids],
                        dtype=torch.long,
                        device=device,
                    )

                    outputs = model(
                        input_ids=input_tensor,
                        output_attentions=True,
                    )

                    # Average over attention heads from the final layer
                    avg_heads = outputs.attentions[-1].mean(dim=1).squeeze(0)

                    # CLS token attention to all subsequent tokens
                    cls_attn = avg_heads[0, 1:].cpu().numpy()

                except RuntimeError as error:
                    if "out of memory" in str(error).lower():
                        print(
                            f"Skipping cell {i} due to OOM "
                            f"(sequence length: {seq_len})"
                        )

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        gc.collect()
                        continue

                    raise

                finally:
                    del input_tensor

                    if outputs is not None:
                        del outputs

                    if avg_heads is not None:
                        del avg_heads

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    gc.collect()

                # Map token IDs to genes
                for gene_tok, attention_score in zip(
                    input_ids[1:],
                    cls_attn,
                ):
                    gene = id_to_gene.get(gene_tok)

                    if gene is not None:
                        all_attn.setdefault(gene, []).append(
                            attention_score
                        )

        # Store mean attention scores for the current cell type
        results_by_cell_type[cell_type] = {
            gene: float(np.mean(scores))
            for gene, scores in all_attn.items()
        }

    print("Done")

    return results_by_cell_type



