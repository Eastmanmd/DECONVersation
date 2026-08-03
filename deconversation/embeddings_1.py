import os
import pickle
import shutil
import logging
import tracemalloc

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

try:
    from geneformer import TranscriptomeTokenizer
    from geneformer import perturber_utils as pu
    from geneformer.emb_extractor import get_embs
    print("geneformer successfully imported.")
except ImportError as e:
    print("geneformer is not installed. Skipping related functions.")
    TranscriptomeTokenizer = None
    pu = None
    get_embs = None

try:
    import cell2sentence as cs
    from cell2sentence.tasks import embed_cells
    from typing import List, Optional
    import time
    print("cell2sentence successfully imported.")
except ImportError:
    print("cell2sentence is not installed. Skipping related functions.")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from llamafactory.model import load_config, load_model, load_tokenizer
    from llamafactory.hparams import get_eval_args, get_infer_args, get_train_args
    from llamafactory.data import get_template_and_fix_tokenizer
    from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence, Tuple, Union
    from transformers import GenerationConfig
    from llamafactory.extras.misc import get_logits_processor
    from llamafactory.chat.base_engine import Response
    import joblib
    import json
    import re
    print("cellhermes successfully imported.")
except ImportError:
    print("cellhermes is not installed. Skipping related functions.")

try:
    import scgpt as scg
    print("scGPT successfully imported.")
except ImportError:
    print("scGPT is not installed. Skipping related functions.")

try:
    import scvi
    print("scVI successfully imported.")
except ImportError:
    print("scVI is not installed. Skipping related functions.")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def extract_embs(
    bulk_df,
    mode,
    temp_output_dir,
    model_path,
    delete_temp_files=False,
    
    # Geneformer only 
    layer_to_quant=17,
    token_output_name="gf_tokens",

    # Cell2Sentence only
    c2s_save_dir=None,
    c2s_save_name="c2s_object",
    model_save_dir=None,
    model_save_name="c2s_model",
    transpose=False,
    gene_name_rm=r"\..+",
    use_genes=None,
    gene_name=None,
    reorder_obs_name=False,
    n_genes=200,
    log=False,
    log_path=None,
    gene_median_file="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/gene_median_dictionary_gc95M.pkl",
    token_dictionary_file="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/token_dictionary_gc95M.pkl",
    gene_mapping_file="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/ensembl_mapping_dict_gc95M.pkl",
):
    safe_temp_dir = os.path.join(temp_output_dir, "temp")
    os.makedirs(safe_temp_dir, exist_ok=True)

    if mode == "geneformer":
        emb = get_embedding_gf(
            bulk_df=bulk_df,
            token_output_dir=safe_temp_dir,
            token_output_name=token_output_name,
            geneformer_model_path=model_path,
            delete_temp_files=delete_temp_files,
            gene_median_file=gene_median_file,
            token_dictionary_file=token_dictionary_file,
            gene_mapping_file=gene_mapping_file,
            layer_to_quant=layer_to_quant,
        )
    elif mode == "c2s":
        emb = get_embedding_c2s(
            bulk_df=bulk_df,
            c2s_save_dir=c2s_save_dir or safe_temp_dir,
            c2s_save_name=c2s_save_name,
            model_path=model_path,
            model_save_dir=model_save_dir or safe_temp_dir,
            model_save_name=model_save_name,
            delete_temp_files=delete_temp_files,
            transpose=transpose,
            gene_name_rm=gene_name_rm,
            use_genes=use_genes,
            gene_name=gene_name,
            reorder_obs_name=reorder_obs_name,
            n_genes=n_genes,
            log=log,
            log_path=log_path,
        )
    elif mode == "cellhermes":
        emb = get_embedding_ch(bulk_df=bulk_df, model_path=model_path)
    elif mode == "scgpt":
        emb = get_embedding_scgpt(bulk_df=bulk_df, model_path=model_path)
    elif mode == "scvi":
        emb = get_embedding_scvi(bulk_df=bulk_df, model_path=model_path)
    else:
        raise ValueError("mode must be 'geneformer', 'c2s', 'cellhermes', 'scgpt' or 'scvi' ")

    return emb


def extract_components(
    bulk_df,
    sig_mat,
    mode,
    transform=True,
    n_components=50,
    layer_to_quant=17,
    token_output_dir=None,
    token_output_name=None,
    delete_temp_files=False,
    geneformer_model_path=None,
    gene_median_file="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/gene_median_dictionary_gc95M.pkl",
    token_dictionary_file="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/token_dictionary_gc95M.pkl",
    gene_mapping_file="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/ensembl_mapping_dict_gc95M.pkl",
    c2s_save_dir=None,
    c2s_save_name="c2s_object",
    model_path=None,
    model_save_dir=None,
    model_save_name="c2s_model",
    transpose=False,
    gene_name_rm=r"\..+",
    use_genes=None,
    gene_name=None,
    reorder_obs_name=False,
    n_genes=200,
    log=False,
    log_path=None,
):
    if mode == "pca":
        dims = get_embedding_pca(
            bulk_df,
            sig_mat,
            transform=transform,
            n_components=n_components,
        )
        return dims

    if mode == "geneformer":
        if token_output_dir is None or token_output_name is None or geneformer_model_path is None:
            raise ValueError(
                "Geneformer mode requires token_output_dir, token_output_name, and geneformer_model_path."
            )

        return get_embedding_gf(
            bulk_df=bulk_df,
            token_output_dir=token_output_dir,
            token_output_name=token_output_name,
            delete_temp_files=delete_temp_files,
            geneformer_model_path=geneformer_model_path,
            gene_median_file=gene_median_file,
            token_dictionary_file=token_dictionary_file,
            gene_mapping_file=gene_mapping_file,
            layer_to_quant=layer_to_quant,
        )

    if mode == "c2s":
        return get_embedding_c2s(
            bulk_df=bulk_df,
            c2s_save_dir=c2s_save_dir,
            c2s_save_name=c2s_save_name,
            model_path=model_path,
            model_save_dir=model_save_dir,
            model_save_name=model_save_name,
            transpose=transpose,
            gene_name_rm=gene_name_rm,
            use_genes=use_genes,
            gene_name=gene_name,
            reorder_obs_name=reorder_obs_name,
            n_genes=n_genes,
            log=log,
            log_path=log_path,
            delete_temp_files=delete_temp_files,
        )

    if mode == "cellhermes":
        return get_embedding_ch(bulk_df=bulk_df, model_path=model_path)

    if mode == "scgpt":
        return get_embedding_scgpt(bulk_df=bulk_df, model_path=model_path)

    if mode == "scvi":
        return get_embedding_scvi(bulk_df=bulk_df, model_path=model_path)

    raise ValueError("mode must be 'pca', 'geneformer', 'c2s', 'cellhermes', 'scgpt' or 'scvi'")


def get_embedding_gf(
    bulk_df,
    token_output_dir,
    token_output_name,
    delete_temp_files,
    geneformer_model_path,
    gene_median_file="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/gene_median_dictionary_gc95M.pkl",
    token_dictionary_file="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/token_dictionary_gc95M.pkl",
    gene_mapping_file="/gpfs/commons/groups/compbio/projects/rf_projects/rf_models/geneformer_pkl/ensembl_mapping_dict_gc95M.pkl",
    layer_to_quant=17,
):
    """
    Generate Geneformer embeddings from a bulk CSV matrix.
    """
    if TranscriptomeTokenizer is None or pu is None or get_embs is None:
        raise ImportError("Geneformer dependencies are not available.")

    if not all(col.startswith("ENSG") for col in bulk_df.columns):
        raise ValueError(
            "Input CSV columns must be Ensembl gene IDs (e.g., ENSG00000123456). "
            "Detected non-Ensembl column names."
        )

    pb_adata = sc.AnnData(bulk_df)
    pb_adata.obs["cell_type"] = "unknown"
    pb_adata.obs["n_counts"] = np.sum(pb_adata.X, axis=1).tolist()
    pb_adata.var["ensembl_id"] = pb_adata.var_names
    pb_adata.X = sp.csc_matrix(pb_adata.X)

    out_adata_path = os.path.join(token_output_dir, f"{token_output_name}.h5ad")
    os.makedirs(token_output_dir, exist_ok=True)
    pb_adata.write_h5ad(out_adata_path)

    print(f"Pseudobulk AnnData saved to: {out_adata_path}")
    print("Starting Geneformer tokenization...")

    tk = TranscriptomeTokenizer(
        {"cell_type": "cell_type"},
        model_input_size=4096,
        special_token=True,
        chunk_size=512,
        gene_median_file=gene_median_file,
        token_dictionary_file=token_dictionary_file,
        gene_mapping_file=gene_mapping_file,
    )

    tk.tokenize_data(
        os.path.dirname(out_adata_path),
        token_output_dir,
        token_output_name,
        file_format="h5ad",
    )

    print("Loading Geneformer model...")
    model = pu.load_model(
        model_type="Pretrained",
        num_classes=0,
        model_directory=geneformer_model_path,
        mode="eval",
    )

    with open(token_dictionary_file, "rb") as f:
        gene_token_dict = pickle.load(f)

    token_gene_dict = {v: k for k, v in gene_token_dict.items()}
    pad_token_id = gene_token_dict.get("<pad>")

    print("Loading tokenized dataset...")

    filtered_input_data = pu.load_and_filter(
        filter_data=None,
        nproc=1,
        input_data_file=f"{token_output_dir}/{token_output_name}.dataset/",
    )

    print("Extracting Geneformer embeddings...")
    state_embs_dict = get_embs(
        model,
        filtered_input_data,
        emb_mode="cell",
        layer_to_quant=layer_to_quant,
        pad_token_id=pad_token_id,
        token_gene_dict=token_gene_dict,
        special_token=True,
        forward_batch_size=50,
    )

    embeddings_df = pd.DataFrame(state_embs_dict.cpu().numpy())
    embeddings_df.index = bulk_df.index
    embeddings_df.columns = "GF_" + embeddings_df.columns.astype(str)

    if delete_temp_files:
        for filename in os.listdir(token_output_dir):
            file_path = os.path.join(token_output_dir, filename)

            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        shutil.rmtree(token_output_dir, ignore_errors=True)

    return embeddings_df


def get_embedding_c2s(
    bulk_df,
    c2s_save_dir,
    c2s_save_name,
    model_path,
    model_save_dir,
    model_save_name,
    transpose=False,
    gene_name_rm=r"\..+",
    use_genes=None,
    gene_name=None,
    reorder_obs_name=False,
    n_genes=200,
    log=False,
    log_path=None,
    delete_temp_files=False,
):
    if log:
        if log_path is None:
            raise ValueError("log=True but no log_path was provided. Please specify a valid log file path.")

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

    if transpose:
        bulk_df = bulk_df.T

    if gene_name_rm is not None:
        bulk_df.columns = bulk_df.columns.str.replace(gene_name_rm, "", regex=True)

    if use_genes is not None:
        missing = set(use_genes) - set(bulk_df.columns)
        if missing:
            raise ValueError(f"Some requested genes not found: {missing}")
        bulk_df = bulk_df[use_genes]

    bulk_df = bulk_df.loc[:, ~bulk_df.columns.duplicated()]

    adata = sc.AnnData(bulk_df)

    if gene_name is not None:
        adata.var_names = gene_name

    adata.obs["organism"] = "Homo sapiens"
    adata.obs["cell_type"] = "unknown"
    adata.obs["tissue"] = "unknown"
    adata.obs["sex"] = "unknown"
    adata.obs["batch_condition"] = "unknown"

    label_cols = ["organism", "cell_type", "tissue", "sex", "batch_condition"]

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
        save_name=c2s_save_name + str(int(time.time() * 1000)),
        dataset_backend="arrow",
    )

    csmodel = cs.CSModel(
        model_name_or_path=model_path,
        save_dir=model_save_dir,
        save_name=model_save_name,
    )

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
    embeddings_df.columns = "C2S_" + embeddings_df.columns.astype(str)

    if log:
        logger.info("Generating embedding complete")
        snapshot = tracemalloc.take_snapshot()
        logger.info(f"Memory snapshot collected")

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            logger.info(f"GPU Memory used: {(total - free) / 1024 / 1024:.2f} MiB")

        logger.info("Pipeline complete")

    if delete_temp_files:
        for filename in os.listdir(c2s_save_dir):
            file_path = os.path.join(c2s_save_dir, filename)

            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        shutil.rmtree(c2s_save_dir, ignore_errors=True)

    return embeddings_df


def get_embedding_ch(bulk_df, model_path):
    args = {
        "model_name_or_path": f"{model_path}",
        "finetuning_type": "lora",
        "template": "llama3",
        "infer_dtype": "float16",
        "do_sample": True,
        "max_new_tokens": 512,
        "temperature": 0.95,
        "top_p": 0.7,
        "top_k": 50,
    }

    model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module['tokenizer']
    processor = tokenizer_module['processor']
    can_generate = finetuning_args.stage == "sft"
    tokenizer.padding_side = "left" if can_generate else "right"
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(
        tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=(not can_generate)
    )
    generating_args = generating_args.to_dict()

    messages = []
    for n in range(0, bulk_df.shape[0]):
        temp_ord = bulk_df.iloc[n, :].sort_values(ascending=False)
        matches = [gene for gene in temp_ord.index if not re.compile("^MT-|^RPL|^RPS").search(gene)]
        prompt = "A cell with genes ranked by expression: " + " ".join(temp_ord[matches][0:500].index.to_list())
        print(prompt)
        messages.append([{"role": "user", "content": f"{prompt}"}])

    generate_outputs = []
    with torch.no_grad():
        for m in messages:
            gen_kwargs, prompt_length = ch_process_args(model, tokenizer, processor, template, generating_args, m, input_kwargs={})
            generate_output = model(gen_kwargs['inputs'], output_hidden_states=True)
            generate_outputs.append(generate_output['hidden_states'][-1].cpu())
    last_hidden_last_word_embs = [token_embs[:, -1, :] for token_embs in generate_outputs]

    colnames = ["CH_" + str(x) for x in range(0, 4096)]
    embeddings_df = pd.DataFrame(columns=colnames)
    for emb1 in last_hidden_last_word_embs:
        embeddings_df.loc[len(embeddings_df)] = emb1[0].tolist()

    embeddings_df.index = bulk_df.index
    return embeddings_df


def ch_process_args(
    model,
    tokenizer,
    processor,
    template,
    generating_args,
    messages,
    input_kwargs,
    system=None,
):
    mm_input_dict = {"images": [], "videos": [], "imglens": [0], "vidlens": [0]}
    messages = template.mm_plugin.process_messages(
        messages, mm_input_dict["images"], mm_input_dict["videos"], processor
    )
    paired_messages = messages + [{"role": "assistant", "content": ""}]
    system = system or generating_args["default_system"]
    prompt_ids, _ = template.encode_oneturn(tokenizer, paired_messages, system)
    prompt_ids, _ = template.mm_plugin.process_token_ids(
        prompt_ids, None, mm_input_dict["images"], mm_input_dict["videos"], tokenizer, processor
    )
    prompt_length = len(prompt_ids)
    inputs = torch.tensor([prompt_ids], device=model.device)
    attention_mask = torch.ones_like(inputs, dtype=torch.bool)

    do_sample: Optional[bool] = input_kwargs.pop("do_sample", None)
    temperature: Optional[float] = input_kwargs.pop("temperature", None)
    top_p: Optional[float] = input_kwargs.pop("top_p", None)
    top_k: Optional[float] = input_kwargs.pop("top_k", None)
    num_return_sequences: int = input_kwargs.pop("num_return_sequences", 1)
    repetition_penalty: Optional[float] = input_kwargs.pop("repetition_penalty", None)
    length_penalty: Optional[float] = input_kwargs.pop("length_penalty", None)
    max_length: Optional[int] = input_kwargs.pop("max_length", None)
    max_new_tokens: Optional[int] = input_kwargs.pop("max_new_tokens", None)
    stop: Optional[Union[str, List[str]]] = input_kwargs.pop("stop", None)
    if stop is not None:
        logger.warning_rank0("Stop parameter is not supported by the huggingface engine yet.")
    generating_args = generating_args.copy()
    generating_args.update(
        dict(
            do_sample=do_sample if do_sample is not None else generating_args["do_sample"],
            temperature=temperature if temperature is not None else generating_args["temperature"],
            top_p=top_p if top_p is not None else generating_args["top_p"],
            top_k=top_k if top_k is not None else generating_args["top_k"],
            num_return_sequences=num_return_sequences,
            repetition_penalty=repetition_penalty
            if repetition_penalty is not None
            else generating_args["repetition_penalty"],
            length_penalty=length_penalty if length_penalty is not None else generating_args["length_penalty"],
            eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
    )

    if isinstance(num_return_sequences, int) and num_return_sequences > 1:
        generating_args["do_sample"] = True
        generating_args["temperature"] = generating_args["temperature"] or 1.0

    if not generating_args["temperature"]:
        generating_args["do_sample"] = False

    if not generating_args["do_sample"]:
        generating_args.pop("temperature", None)
        generating_args.pop("top_p", None)

    if max_length:
        generating_args.pop("max_new_tokens", None)
        generating_args["max_length"] = max_length

    if max_new_tokens:
        generating_args.pop("max_length", None)
        generating_args["max_new_tokens"] = max_new_tokens

    gen_kwargs = dict(
        inputs=inputs,
        attention_mask=attention_mask,
        generation_config=GenerationConfig(**generating_args),
        logits_processor=get_logits_processor(),
    )
    mm_inputs = template.mm_plugin.get_mm_inputs(**mm_input_dict, seqlens=[prompt_ids], processor=processor)
    for key, value in mm_inputs.items():
        value = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        gen_kwargs[key] = value.to(model.device)

    return gen_kwargs, prompt_length


def get_embedding_scgpt(bulk_df, model_path):
    adata = sc.AnnData(bulk_df)
    adata.var["gene_name"] = adata.var.index
    adata.obs["sample"] = adata.obs.index

    embed_adata = scg.tasks.embed_data(
        adata,
        model_path,
        gene_col="gene_name",
        obs_to_save="sample",
        batch_size=64,
        return_new_adata=True,
    )
    embeddings_df = pd.DataFrame(embed_adata.X)
    embeddings_df.columns = "scGPT_" + embeddings_df.columns.astype(str)
    embeddings_df["name"] = adata.obs_names.to_list()
    embeddings_df = embeddings_df.set_index("name")
    return embeddings_df


def get_embedding_scvi(bulk_df, model_path):
    adata = sc.AnnData(bulk_df)
    adata.obs_names = bulk_df.index
    adata.var_names = bulk_df.columns
    adata.obs["batch"] = "bulk"
    adata.obs["id"] = bulk_df.index

    scvi.model.SCVI.prepare_query_anndata(adata, model_path)
    vae_q = scvi.model.SCVI.load_query_data(adata, model_path)
    vae_q.is_trained = True

    embeddings = vae_q.get_latent_representation()
    embeddings_df = pd.DataFrame(embeddings, index=adata.obs_names)
    embeddings_df.columns = "scVI_" + embeddings_df.columns.astype(str)

    return embeddings_df


def get_embedding_pca(bulk_df, sig_mat, transform=True, n_components=50):
    shared_genes = bulk_df.columns.intersection(sig_mat.columns)
    bulk = bulk_df[shared_genes].copy()
    sig = sig_mat[shared_genes].copy()

    if transform:
        bulk = np.log1p(bulk)
        sig = np.log1p(sig)

    scaler = StandardScaler()
    sig_scaled = scaler.fit_transform(sig)

    n_components = min(n_components, *sig_scaled.shape)
    pca = PCA(n_components=n_components)
    pca.fit(sig_scaled)

    pc_cols = [f"PC_{i+1}" for i in range(n_components)]
    sig_pca = pd.DataFrame(pca.transform(sig_scaled), index=sig.index, columns=pc_cols)
    bulk_pca = pd.DataFrame(pca.transform(scaler.transform(bulk)), index=bulk.index, columns=pc_cols)

    print(f"Variance explained by {n_components} PCs: {pca.explained_variance_ratio_.sum():.2%}")

    return {
        "pca_bulk": bulk_pca,
        "sig_pca": sig_pca,
    }
