import gc
import os
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import scanpy as sc
import anndata as ad
from datasets import load_from_disk
from transformers import AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from peft import PeftModel
import json
from preprocessing import gene_id_name_map


# ===============================
# Geneformer
# ===============================
try:
    from geneformer import TranscriptomeTokenizer
    print("geneformer successfully imported.")
    
except ImportError as e:
    print("geneformer is not installed. Skipping related functions.")



# ---------------------------------
# Attention extractor
# ---------------------------------
def get_attention(
    data,
    mode,  # "cellhermes", "c2s" or "geneformer"
    model_path,
    cell_types,
    cell_type_col="cell_type",
    num_of_cells=50,
    random_state=42,
    
    # cellHermes / C2S only
    n_genes=1000,
    
    # Geneformer only
    output_dir=None,
    output_name="gf_attention",
    model_version="V2",
    min_cells_per_cell_type=None,
):
    # Get shared parameters
    mode = str(mode).lower()
    common = dict(
        data=data,
        model_path=model_path,
        cell_types=cell_types,
        cell_type_col=cell_type_col,
        num_of_cells=num_of_cells,
        random_state=random_state,
    )

    # cellHermes
    if mode == "cellhermes":
        return get_attention_by_gene_ch(
            **common,
            n_genes = n_genes
        )
        
    # cell2sentence 
    if mode == "c2s":
        return get_attention_by_gene_c2s(
            **common,
            n_genes = n_genes
        )   

    # Geneformer
    if mode == "geneformer":
        if output_dir is None:
            raise ValueError("output_dir is required when model_type='geneformer'")
        if n_genes is not None:
            print("Note: n_genes is ignored for model_type='geneformer'.")
        return get_attention_by_gene_gf(
            **common,
            output_dir=output_dir,
            output_name=output_name,
            model_version=model_version,
            min_cells_per_cell_type=min_cells_per_cell_type,
        )

    raise ValueError(f"Unknown model_type '{model_type}'. Expected 'cellhermes', 'c2s' or 'geneformer' ")



# ---------------------------------
# Extract C2S attention 
# ---------------------------------
def _build_symbol_map(ensembl_ids):
    """Map Ensembl IDs to gene symbols in ONE call, falling back to the ID."""
    ensembl_ids = list(ensembl_ids)
    mapped = gene_id_name_map(ensembl_ids, mode="to_symbol")

    # Support mappers that return either a dict or a list aligned with input
    if isinstance(mapped, dict):
        return {eid: (mapped.get(eid) or eid) for eid in ensembl_ids}
    return {eid: (sym or eid) for eid, sym in zip(ensembl_ids, mapped)}


# ---------------------------------
# Extract geneformer attention 
# ---------------------------------
def get_attention_by_gene_gf(
    data,  # anndata or dataframe
    model_path,
    output_dir,
    cell_types,  # Accepts a single string or a list of cell types
    output_name="gf_attention",
    cell_type_col="cell_type",
    model_version="V2",
    num_of_cells=50,
    random_state=42,
    min_cells_per_cell_type=None,
):
    """
    Extract per-gene attention scores from Geneformer for one or more cell types.

    Runs cells through a pretrained or fine-tuned Geneformer model and extracts the
    CLS token's attention over gene tokens from the final layer, maps token IDs to
    gene symbols, and averages across cells.

    Returns
    -------
    dict
        {cell_type: {gene_symbol: mean_attention_score, ...}, ...}
    """

    # Ensure cell_types is a list even if a single string is passed
    if isinstance(cell_types, str):
        cell_types = [cell_types]

    # ------------ Prepare AnnData ---------------------------------
    if isinstance(data, pd.DataFrame):
        # Convert a DataFrame (e.g. signature matrix) to AnnData
        data_df = data.copy()
        data = sc.AnnData(data_df)
        data.obs[cell_type_col] = data_df.index.astype(str)
        data.X = sp.csr_matrix(data.X)

    elif isinstance(data, ad.AnnData):
        if min_cells_per_cell_type is not None:
            subsampled_index = (
                data.obs.groupby(cell_type_col, group_keys=False, observed=True)
                .apply(
                    lambda group: group.sample(
                        n=min(len(group), min_cells_per_cell_type),
                        random_state=random_state,
                    )
                )
                .index
            )
            data = data[subsampled_index].copy()
            print(f"Subsetted data shape: {data.shape}")
        else:
            data = data.copy()
    else:
        raise TypeError("data must be an AnnData object or a pandas DataFrame")

    # Geneformer requires these fields
    if "n_counts" not in data.obs:
        data.obs["n_counts"] = np.asarray(data.X.sum(axis=1)).ravel()
    if "ensembl_id" not in data.var:
        data.var["ensembl_id"] = data.var_names

    # FIX: write the h5ad into its own input folder and tokenize THAT folder
    # (previously the parent of output_dir was tokenized by mistake)
    input_dir = os.path.join(output_dir, "input_h5ad")
    os.makedirs(input_dir, exist_ok=True)
    data.write_h5ad(os.path.join(input_dir, f"{output_name}_data.h5ad"))

    # ------------ Tokenize data -----------------------------------
    print("Starting tokenization...")
    tk = TranscriptomeTokenizer(
        {cell_type_col: "cell_type"},  # FIX: respect cell_type_col
        model_input_size=4096,
        special_token=True,
        chunk_size=512,
        model_version=model_version,
    )
    tk.tokenize_data(input_dir, output_dir, "tokenized", file_format="h5ad")
    print("Tokenization done.")

    dataset_path = os.path.join(output_dir, "tokenized.dataset")
    dataset = load_from_disk(dataset_path)

    # ------------ Gene ID -> symbol mapping (done ONCE) -----------
    # FIX: previously gene_id_name_map was called for every token of every cell
    id_to_gene = {token_id: gene for gene, token_id in tk.gene_token_dict.items()}
    genes_in_data = set(data.var["ensembl_id"].astype(str))
    ids_to_map = [g for g in id_to_gene.values() if g in genes_in_data]
    print(f"Mapping {len(ids_to_map)} Ensembl IDs to gene symbols...")
    ensembl_to_symbol = _build_symbol_map(ids_to_map)

    # ------------ Load model --------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("Warning: running on CPU; long sequences will be slow.")

    model = AutoModel.from_pretrained(model_path, output_attentions=True)
    model.eval()
    model.to(device)

    results_by_cell_type = {}

    # ------------ Extract attention per cell type -----------------
    for cell_type in cell_types:
        # input_columns avoids loading full rows just to check the label
        filtered_dataset = dataset.filter(
            lambda ct: ct == cell_type, input_columns="cell_type"
        )

        if len(filtered_dataset) == 0:
            print(f"No cells found for cell type '{cell_type}', skipping.")
            continue

        num_cells_to_process = min(num_of_cells, len(filtered_dataset))
        subset = filtered_dataset.shuffle(seed=random_state).select(
            range(num_cells_to_process)
        )

        all_attn = {}

        with torch.inference_mode():
            print(f"Processing {len(subset)} cells for cell type '{cell_type}'...")
            for i, sample in enumerate(subset):
                input_ids = sample["input_ids"]
                seq_len = len(input_ids)
                if seq_len <= 1:
                    continue

                try:
                    input_tensor = torch.tensor(
                        [input_ids], dtype=torch.long, device=device
                    )
                    outputs = model(input_ids=input_tensor, output_attentions=True)

                    # Final layer, averaged over heads; CLS row -> gene tokens
                    cls_attn = (
                        outputs.attentions[-1][0].mean(dim=0)[0, 1:].float().cpu().numpy()
                    )
                    del outputs, input_tensor

                except RuntimeError as error:
                    if "out of memory" in str(error).lower():
                        print(f"Skipping cell {i} due to OOM (sequence length: {seq_len})")
                        # FIX: only clear caches when actually needed
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    raise

                # Map token IDs -> Ensembl IDs -> symbols (cheap dict lookups)
                for gene_tok, attention_score in zip(input_ids[1:], cls_attn):
                    gene = id_to_gene.get(gene_tok)
                    if gene is None:
                        continue  # special tokens such as <eos>/<pad>
                    symbol = ensembl_to_symbol.get(gene, gene)
                    all_attn.setdefault(symbol, []).append(float(attention_score))

        results_by_cell_type[cell_type] = {
            gene: float(np.mean(scores)) for gene, scores in all_attn.items()
        }

    print("Done")
    return results_by_cell_type



# ---------------------------------
# Extract C2S attention 
# ---------------------------------
def get_attention_by_gene_c2s(
    data,
    model_path,
    cell_types,
    cell_type_col="cell_type",
    num_of_cells=50,
    random_state=42,
    n_genes=500,
):
    """
    Extract per-gene attention scores from Cell2Sentence for one or more cell types.

    Runs cells through a pretrained or fine-tuned C2S-Scale-Gemma model and extracts
    the final prompt token's attention over gene tokens from the last layer, then
    maps token spans back to gene names and averages across cells. The result ranks
    genes by how much the model attends to them when representing each cell type.

    C2S is a causal decoder-only model, so unlike Geneformer's bidirectional CLS
    token, there is no dedicated summary token to read attention from. The last
    prompt token (the position immediately preceding cell-type generation) plays
    the same role here, since it is the position whose attention distribution
    actually drives the model's prediction.

    Parameters
    ----------
    data : anndata.AnnData
        Expression data. Gene identifiers must be gene symbols, as expected by C2S's
        cell-sentence construction.
    model_path : str
        Path to C2S model
    cell_types : str or list of str
        Cell type(s) to extract attention for. A single string is accepted. Each
        type is filtered and processed separately.
    cell_type_col : str, default "cell_type"
        Column in ``data.obs`` holding the cell type labels matched against
        ``cell_types``.
    num_of_cells : int, default 50
        Maximum cells to process per cell type. The effective count is
        ``min(num_of_cells, n_available)``, so fewer are used when a type is rare.
    random_state : int, default 42
        Seed for subsampling, for reproducible cell selection.
    n_genes : int, default 500
        Number of top-expressed genes to include per cell sentence. Attention cost
        scales roughly quadratically with this value.

    Returns
    -------
    dict
        Nested dictionary mapping each cell type to its per-gene mean attention
        scores::

            {cell_type: {gene_name: mean_attention_score, ...}, ...}

        Scores are averaged over the cells processed for that type. Cell types with
        no matching cells are absent from the result.
    """
    
    if isinstance(cell_types, str):
        cell_types = [cell_types]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # eager attention required: SDPA/flash-attention backends never
    # materialize the full attention matrix, so output_attentions is
    # silently ignored (or errors) under the default backend.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        base_model_name = json.load(
            open(os.path.join(model_path, "adapter_config.json"))
        )["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
    exclude_re = re.compile(r"^MT-|^RPL|^RPS")
    results_by_cell_type = {}

    for cell_type in cell_types:
        mask = data.obs[cell_type_col] == cell_type
        n_available = int(mask.sum())
        subset = (
            data.obs[mask]
            .sample(n=min(num_of_cells, n_available), random_state=random_state)
            .index
        )

        print(f"Processing {len(subset)} cells for cell type '{cell_type}'...")

        all_attn = {}

        with torch.inference_mode():
            for cell_idx in subset:
                row = data.obs.index.get_loc(cell_idx)
                expr = data.X[row]
                expr = expr.toarray().squeeze() if hasattr(expr, "toarray") else expr

                ranked = pd.Series(expr, index=data.var_names).sort_values(ascending=False)
                genes_ranked = [g for g in ranked.index if not exclude_re.search(g)][:n_genes]
                cell_sentence = " ".join(genes_ranked)

                prompt = (
                    f"The following is a list of {len(genes_ranked)} gene names ordered by "
                    f"descending expression level in a Homo sapiens cell. Your task is to give "
                    f"the cell type which this cell belongs to based on its gene expression.\n"
                    f"Cell sentence: {cell_sentence}.\n"
                    f"The cell type corresponding to these genes is:"
                )

                enc = None
                outputs = None
                avg_heads = None

                try:
                    enc = tokenizer(
                        prompt, return_tensors="pt", return_offsets_mapping=True
                    ).to(device)
                    offsets = enc.pop("offset_mapping")[0].tolist()

                    outputs = model(**enc, output_attentions=True)

                    # Average over heads from the final layer
                    avg_heads = outputs.attentions[-1].mean(dim=1).squeeze(0)

                    # Last prompt token's attention over everything before it
                    last_attn = avg_heads[-1, :-1].float().cpu().numpy()

                except RuntimeError as error:
                    if "out of memory" in str(error).lower():
                        print(f"Skipping a cell due to OOM (prompt tokens: {len(offsets) if 'offsets' in dir() else '?'})")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    raise

                finally:
                    del enc, outputs, avg_heads
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                # Map token spans back to genes
                cursor = prompt.index(cell_sentence)
                for gene in genes_ranked:
                    g_start = prompt.index(gene, cursor)
                    g_end = g_start + len(gene)
                    cursor = g_end
                    idxs = [i for i, (s, e) in enumerate(offsets[:-1]) if s < g_end and e > g_start]
                    if idxs:
                        all_attn.setdefault(gene, []).append(float(np.mean(last_attn[idxs])))

        results_by_cell_type[cell_type] = {
            gene: float(np.mean(scores)) for gene, scores in all_attn.items()
        }

    print("Done")
    return results_by_cell_type


# ---------------------------------
# Extract cellHermes attention 
# ---------------------------------
def get_attention_by_gene_ch(
    data, 
    model_path, 
    cell_types, 
    cell_type_col="cell_type",
    num_of_cells=50, 
    random_state=42, 
    n_genes=1000):
    
    """
    Extract per-gene attention scores from cellHermes for one or more cell types.

    Parameters
    ----------
    data : anndata.AnnData
        Expression data. Gene identifiers must be gene symbols, as expected by C2S's
        cell-sentence construction.
    model_path : str
        Path to C2S model
    cell_types : str or list of str
        Cell type(s) to extract attention for. A single string is accepted. Each
        type is filtered and processed separately.
    cell_type_col : str, default "cell_type"
        Column in ``data.obs`` holding the cell type labels matched against
        ``cell_types``.
    num_of_cells : int, default 50
        Maximum cells to process per cell type. The effective count is
        ``min(num_of_cells, n_available)``, so fewer are used when a type is rare.
    random_state : int, default 42
        Seed for subsampling, for reproducible cell selection.
    n_genes : int, default 500
        Number of top-expressed genes to include per cell sentence. Attention cost
        scales roughly quadratically with this value.

    Returns
    -------
    dict
        Nested dictionary mapping each cell type to its per-gene mean attention
        scores::

            {cell_type: {gene_name: mean_attention_score, ...}, ...}

        Scores are averaged over the cells processed for that type. Cell types with
        no matching cells are absent from the result.
    """
    if isinstance(cell_types, str):
        cell_types = [cell_types]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, output_attentions=True, attn_implementation="eager",
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    results_by_cell_type = {}

    for cell_type in cell_types:
        subset = (data.obs[data.obs[cell_type_col] == cell_type]
                  .sample(n=min(num_of_cells, (data.obs[cell_type_col] == cell_type).sum()),
                          random_state=random_state).index)

        all_attn = {}
        with torch.inference_mode():
            for cell_idx in subset:
                expr = data.X[data.obs.index.get_loc(cell_idx)]
                expr = expr.toarray().squeeze() if hasattr(expr, "toarray") else expr
                ranked = pd.Series(expr, index=data.var_names).sort_values(ascending=False)
                genes_ranked = [g for g in ranked.index if not re.compile("^MT-|^RPL|^RPS").search(g)][:n_genes]
                cell_sentence = " ".join(genes_ranked)
                prompt = f"A cell with genes ranked by expression: {cell_sentence}."

                enc = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True).to(device)
                offsets = enc.pop("offset_mapping")[0].tolist()

                outputs = model(**enc, output_attentions=True)
                avg_heads = outputs.attentions[-1].mean(dim=1).squeeze(0)  # [seq, seq]
                last_attn = avg_heads[-1, :-1].cpu().numpy()               # last token -> everything before it

                # map each gene's char span back to its token span (same trick as C2S)
                cursor = prompt.index(cell_sentence)
                for gene in genes_ranked:
                    g_start = prompt.index(gene, cursor)
                    g_end = g_start + len(gene)
                    cursor = g_end
                    idxs = [i for i, (s, e) in enumerate(offsets[:-1]) if s < g_end and e > g_start]
                    if idxs:
                        all_attn.setdefault(gene, []).append(float(np.mean(last_attn[idxs])))

        results_by_cell_type[cell_type] = {g: float(np.mean(v)) for g, v in all_attn.items()}

    return results_by_cell_type
    