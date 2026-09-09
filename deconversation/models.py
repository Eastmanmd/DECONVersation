# ===============================
# Standard Libraries
# ===============================
import os
import gc
import pickle
import warnings
import logging
import anndata
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
import scanpy as sc
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ===============================
# For LoRA fine-tuning 
# ===============================
#import torch
#from datasets import load_from_disk
#from transformers import BertForSequenceClassification, TrainingArguments, Trainer, AutoModelForCausalLM, BitsAndBytesConfig
#from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


# ===============================
# Geneformer
# ===============================
try:
    from geneformer import Classifier, TranscriptomeTokenizer
    from geneformer import DataCollatorForCellClassification
    print("geneformer successfully imported.")
    import torch
    from datasets import load_from_disk
    from transformers import BertForSequenceClassification, TrainingArguments, Trainer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
except ImportError:
    print("geneformer is not installed. Skipping related functions.")


# ===============================
# Cell2Sentence
# ===============================
try:
    import cell2sentence as cs
    print("Cell2Sentence successfully imported.")
    import torch
    from datasets import load_from_disk
    from transformers import BertForSequenceClassification, TrainingArguments, Trainer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
except ImportError:
    print("Cell2Sentence is not installed. Skipping related functions.")

# ===============================
# scGPT
# ===============================
try:
    import scgpt as scg
    from scgpt.model import TransformerModel, AdversarialDiscriminator
    from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
    from scgpt.loss import (
        masked_mse_loss,
        masked_relative_error,
        criterion_neg_log_bernoulli,
    )
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.preprocess import Preprocessor
    from scgpt import SubsetsBatchSampler
    from scgpt.utils import set_seed, category_str2int, eval_scib_metrics
    print("scGPT successfully imported.")
    import copy
    import gc
    import json
    import os
    from pathlib import Path
    import shutil
    import sys
    import time
    import traceback
    from typing import List, Tuple, Dict, Union, Optional
    import warnings
    import pandas as pd
    import pickle
    import torch
    from anndata import AnnData
    import scanpy as sc
    import scvi
    import seaborn as sns
    import numpy as np
    import wandb
    from scipy.sparse import issparse
    import matplotlib.pyplot as plt
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from torchtext.vocab import Vocab
    from torchtext._torchtext import (
        Vocab as VocabPybind,
    )
    from sklearn.metrics import confusion_matrix
    sc.set_figure_params(figsize=(6, 6))
    os.environ["KMP_WARNINGS"] = "off"
    warnings.filterwarnings('ignore')
except ImportError:
    print("scGPT is not installed. Skipping related functions.")

# =================================================
#    Full Geneformer Finetuning
# =================================================

# Default training arguments
# -------------------------------------------
DEFAULT_TRAINING_ARGS = {
    "num_train_epochs": 3,
    "learning_rate": 5e-5,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 190,
    "weight_decay": 0.1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "seed": 42,
    "fp16": True,
}


def train_geneformer_cell_classifier(
    adata,
    output_dir,
    state_key,
    cell_types = None, # should be a list
    tokenize = True,
    tokenized_data_dir = None, # path to store tokenized data
    token_output_prefix = "tokenized", #  prefix for tokenized data
    undersample = True, # Should data be undersampled
    n_per_class = 50, #if undersampling, max # cell per cell type
    n_test_per_class = 25, # max number of  cells per cell type in test dataset (holdout dataset)
    eval_fraction = 0.2, # Fraction of test data (hold out dataset)
    model_directory = "ctheodoris/Geneformer", # Model directory
    freeze_layers = 2, # freeze layer
    num_crossval_splits = 1, # single train/eval split, 5 5-fold cross validation
    forward_batch_size = 10, # batch size 
    nproc = 8,
    max_ncells = None, # Max number of cells for fine-tuning 
    training_args = None, # training arguments 
    output_prefix = "gf_finetune", # output directory 
    metrics_output_dir = None, # if seperate output directory is required 
):

    # ------------------------------------------------------------------
    # Validate Directories
    # ------------------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metrics_output_dir is None:
        metrics_output_dir = output_dir / "metrics"
    metrics_output_dir = Path(metrics_output_dir)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    if not tokenize and tokenized_data_dir is None:
        raise ValueError(
            "You must provide `tokenized_data_dir` when `tokenize=False`."
        )

    # ------------------------------------------------------------------
    # Ensure cell_id exists in adata.obs
    # ------------------------------------------------------------------
    if "cell_id" not in adata.obs.columns:
        raise RuntimeError(
            "'cell_id' column not found in adata.obs. "
            "Please run load_and_prep_data() to prepare your data before calling this function."
        )

    if state_key not in adata.obs.columns:
        raise ValueError(
            f"state_key '{state_key}' not found in adata.obs columns: "
            f"{list(adata.obs.columns)}"
        )

    # ------------------------------------------------------------------
    #  Merge training args with defaults
    # ------------------------------------------------------------------
    merged_training_args = {**DEFAULT_TRAINING_ARGS, **(training_args or {})}
    logger.info("Training arguments: %s", merged_training_args)

    # ------------------------------------------------------------------
    #  Tokenize (optional)
    # ------------------------------------------------------------------
    if tokenize:
        token_dir = output_dir / "tokenized"
        token_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Tokenizing data → %s", token_dir)

        tokenizer = TranscriptomeTokenizer(
            custom_attr_name_dict={"cell_id": "cell_id", state_key: state_key},
            nproc=nproc,
        )
        tokenizer.tokenize_data(
            data_directory=str(output_dir),   # adata must be saved here first
            output_directory=str(token_dir),
            output_prefix=token_output_prefix,
            file_format="h5ad",
        )
        tokenized_data_dir = str(token_dir / f"{token_output_prefix}.dataset")
        logger.info("Tokenization complete: %s", tokenized_data_dir)
    else:
        logger.info("Skipping tokenization. Using: %s", tokenized_data_dir)

    # ------------------------------------------------------------------
    #  Build filter dict (cell-type subset)
    # ------------------------------------------------------------------
    # Filter to specific cell types 
    filter_data_dict = None
    if cell_types is not None:
        filter_data_dict = {state_key: cell_types}
        logger.info("Filtering to cell types: %s", cell_types)

    # ------------------------------------------------------------------
    # Build metadata splits
    # ------------------------------------------------------------------
    metadata = adata.obs.copy()

    # --- Training pool (undersample or full) ---
    if undersample:
        logger.info("Undersampling: max %d cells per class", n_per_class)
        pool = (
            metadata
            .groupby(state_key, group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), n_per_class), random_state=merged_training_args.get("seed", 42)))
        )
    else:
        logger.info("Using all available cells for training pool.")
        pool = metadata.copy()

    pool_ids = set(pool.index)

    # --- Test pool: cells NOT in training pool ---
    holdout = metadata[~metadata["cell_id"].isin(pool_ids)]
    test_data = (
        holdout
        .groupby(state_key, group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), n_test_per_class), random_state=merged_training_args.get("seed", 42)))
    )

    logger.info(
        "Split sizes — train pool: %d | test: %d",
        len(pool), len(test_data),
    )

    # --- Train / eval split within pool ---
    train_data, eval_data = train_test_split(
        pool,
        test_size=eval_fraction,
        random_state=merged_training_args.get("seed", 42),
        stratify=pool[state_key],
    )

    logger.info(
        "Train: %d | Eval: %d | Test: %d",
        len(train_data), len(eval_data), len(test_data),
    )

    # ------------------------------------------------------------------
    # Set up Classifier
    # ------------------------------------------------------------------
    cc = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": state_key, "states": "all"},
        filter_data=filter_data_dict,
        training_args=merged_training_args,
        max_ncells=max_ncells,
        freeze_layers=freeze_layers,
        num_crossval_splits=num_crossval_splits,
        forward_batch_size=forward_batch_size,
        nproc=nproc,
    )

    # ------------------------------------------------------------------
    # Prepare data (train+test labeling)
    # ------------------------------------------------------------------
    train_test_id_split_dict = {
        "attr_key": "cell_id",
        "train": list(train_data.index) + list(eval_data.index),
        "test": list(test_data.index),
    }

    logger.info("Preparing data splits…")
    cc.prepare_data(
        input_data_file=tokenized_data_dir,
        output_directory=str(output_dir),
        output_prefix=output_prefix,
        split_id_dict=train_test_id_split_dict,
    )

    # ------------------------------------------------------------------
    # Train (validate) the model
    # ------------------------------------------------------------------
    train_valid_id_split_dict = {
        "attr_key": "cell_id",
        "train": list(train_data.index),
        "eval": list(eval_data.index),
    }

    labeled_train_path = str(output_dir / f"{output_prefix}_labeled_train.dataset/")
    id_class_dict_path = str(output_dir / f"{output_prefix}_id_class_dict.pkl")

    logger.info("Starting training…")
    all_metrics = cc.validate(
        model_directory=model_directory,
        prepared_input_data_file=labeled_train_path,
        id_class_dict_file=id_class_dict_path,
        output_directory=str(output_dir),
        output_prefix=output_prefix,
        split_id_dict=train_valid_id_split_dict,
    )

    # ------------------------------------------------------------------
    # Evaluate on held-out test set
    # ------------------------------------------------------------------
    date_string = datetime.now().strftime("%y%m%d")
    finetuned_model_dir = (
        output_dir
        / f"{date_string}_geneformer_cellClassifier_{output_prefix}"
        / "ksplit1"
    )

    if not finetuned_model_dir.exists():
        # Fallback: find the most recently created matching directory
        candidates = sorted(
            output_dir.glob(f"*_geneformer_cellClassifier_{output_prefix}/ksplit1"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"Could not locate fine-tuned model directory under {output_dir}. "
                "Ensure training completed successfully."
            )
        finetuned_model_dir = candidates[0]
        logger.warning("Using fallback model directory: %s", finetuned_model_dir)

    labeled_test_path = str(output_dir / f"{output_prefix}_labeled_test.dataset/")

    logger.info("Evaluating on test set…")
    all_metrics_test = cc.evaluate_saved_model(
        model_directory=str(finetuned_model_dir),
        id_class_dict_file=id_class_dict_path,
        test_data_file=labeled_test_path,
        output_directory=str(metrics_output_dir),
        output_prefix=output_prefix,
    )

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    metrics_path = metrics_output_dir / f"{output_prefix}_metrics.pkl"
    with open(metrics_path, "wb") as f:
        pickle.dump({"train_metrics": all_metrics, "test_metrics": all_metrics_test}, f)
    logger.info("Metrics saved to %s", metrics_path)

    logger.info("Done.")
    return {"train_metrics": all_metrics, "test_metrics": all_metrics_test}


# =====================================
#    LoRA Geneformer Finetuning
# =====================================
    
def train_geneformer_cell_classifier_LoRA(
    train_dataset, #tokenized training data 
    test_dataset, #tokenized test data 
    class_id_pkl, # .pkl file post tokenization containing ids and labels (path)
    output_dir,
    gene_token_dict_pkl,
    model_save_dir,
    model_name = "ctheodoris/Geneformer",
    training_args = None,
    ):

    # ──────────────── Load tokenized data ────────────────────  
    train_dataset = load_from_disk(train_dataset)
    val_dataset   = load_from_disk(test_dataset)

    # ───────────────Get class ID maps ─────────────────────────
    with open(class_id_pkl, "rb") as f:
        id2label = pickle.load(f) # ID to label
    label2id = {v: k for k, v in id2label.items()} # Label to ID

    # ────────── Load base model with classification head ─────
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True)

    # ─────────────── Wrap with LoRA ─────────────────
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,        # sequence classification
        r=8,                               # rank 
        lora_alpha=16,                     # scaling factor (usually 2x r)
        lora_dropout=0.1,
        target_modules=["query", "value"], # inject into Q and V projections
        bias="none",                       # don't train biases
        modules_to_save=["classifier"],    # always train the classification head
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ──────────────── Compute Metrics ─────────────────
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    # ────────────────── Training arguments ────────────────────
    training_args = TrainingArguments(
        output_dir= output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        learning_rate=5e-4,         
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none")

    # ────────────────── Handle padding ──────────────────
    # Load the gene token dictionary file
    with open(gene_token_dict_pkl, "rb") as f:
        gene_token_dict = pickle.load(f)
    
    # Dynamically pad the gene sequences in each batch
    data_collator = DataCollatorForCellClassification(token_dictionary=gene_token_dict)

    # ────────────────── Trainer ───────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    # ────────────────── Save Model ─────────────────────────
    model.save_pretrained(model_save_dir)


# =====================================================
#    Full Cell2Sentence Finetuning
# =====================================================
try:
    DEFAULT_TRAIN_ARGS_C2S = TrainingArguments(
        bf16=True,
        fp16=False,
        report_to="none",
        prediction_loss_only=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        load_best_model_at_end=True,
        logging_steps=50,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        num_train_epochs=10,
        eval_steps=50,
        eval_strategy="steps",
        save_steps=100,
        save_strategy="steps",
        save_total_limit=3,
        warmup_ratio=0.05,
        output_dir="./tmp_output",  # overwritten at runtime
        torch_empty_cache_steps=1,
    )
except NameError:
    pass

def train_c2s_cell_classifier(
    adata,
    cell_type_col,
    model_path,
    c2s_save_dir,
    c2s_save_name,
    save_dir,
    save_name,
    seed,
    tissue = "unknown",
    batch = None,
    sex_col = None,
    organism = "Homo sapiens",
    top_k_genes = 200,
    max_eval_samples = 500,
    train_args_override = None,
):
    # Merge training args
    if train_args_override:
        base = DEFAULT_TRAIN_ARGS_C2S.to_dict()
        base.update(train_args_override)
        train_args = TrainingArguments(**base)
    else:
        train_args = DEFAULT_TRAIN_ARGS_C2S

    # Add Metadata
    adata.obs["organism"] = organism
    adata.obs["cell_type"] = adata.obs[cell_type_col]
    adata.obs["tissue"] = tissue
    adata.obs["sex"] = adata.obs[sex_col] if sex_col else "unknown"
    adata.obs["batch_condition"] = adata.obs[batch] if batch else "unknown"

    adata_obs_cols_to_keep = ["organism", "cell_type", "tissue", "sex", "batch_condition"]

    # Build C2S dataset
    arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
        adata=adata,
        random_state=seed,
        sentence_delimiter=" ",
        label_col_names=adata_obs_cols_to_keep,
    )
    csdata = cs.CSData.csdata_from_arrow(
        arrow_dataset=arrow_ds,
        vocabulary=vocabulary,
        save_dir=c2s_save_dir,
        save_name=c2s_save_name+str(int(time.time() * 1000)),
        dataset_backend="arrow",
    )

    # Load model and resolve output dir
    csmodel = cs.CSModel(
        model_name_or_path=model_path,
        save_dir=save_dir,
        save_name=save_name,
    )
    training_task = "cell_type_prediction"
    datetimestamp = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    output_dir = os.path.join(csmodel.save_dir, f"{datetimestamp}_finetune_{training_task}")
    os.makedirs(output_dir, exist_ok=True)
    train_args = TrainingArguments(**{**train_args.to_dict(), "output_dir": output_dir})

    # Free memory before training
    del arrow_ds, adata
    torch.cuda.empty_cache()
    gc.collect()

    # Fine-tune
    csmodel.fine_tune(
        csdata=csdata,
        task=training_task,
        train_args=train_args,
        loss_on_response_only=False,
        top_k_genes=top_k_genes,
        max_eval_samples=max_eval_samples,
    )



# =====================================================
#    LoRA Cell2Sentence Finetuning
# =====================================================
def train_c2s_cell_classifier_LoRA(
    adata,
    cell_type_col,
    model_path,
    c2s_save_dir,
    c2s_save_name,
    save_dir,
    save_name,
    seed,
    tissue = "unknown",
    batch = None,
    sex_col = None,
    organism = "Homo sapiens",
    top_k_genes = 200,
    max_eval_samples = 500,
    train_args_override = None,
):
    # Merge training args
    if train_args_override:
        base = DEFAULT_TRAIN_ARGS_C2S.to_dict()
        base.update(train_args_override)
        train_args = TrainingArguments(**base)
    else:
        train_args = DEFAULT_TRAIN_ARGS_C2S

    # Add Metadata
    adata.obs["organism"] = organism
    adata.obs["cell_type"] = adata.obs[cell_type_col]
    adata.obs["tissue"] = tissue
    adata.obs["sex"] = adata.obs[sex_col] if sex_col else "unknown"
    adata.obs["batch_condition"] = adata.obs[batch] if batch else "unknown"

    adata_obs_cols_to_keep = ["organism", "cell_type", "tissue", "sex", "batch_condition"]

    # Build C2S dataset
    arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
        adata=adata,
        random_state=seed,
        sentence_delimiter=" ",
        label_col_names=adata_obs_cols_to_keep,
    )
    csdata = cs.CSData.csdata_from_arrow(
        arrow_dataset=arrow_ds,
        vocabulary=vocabulary,
        save_dir=c2s_save_dir,
        save_name=c2s_save_name+str(int(time.time() * 1000)),
        dataset_backend="arrow",
    )

    # Free memory before loading the model
    del arrow_ds, adata
    torch.cuda.empty_cache()
    gc.collect()

    # ---- MONKEY PATCH FOR LORA INTEGRATION ----
    original_from_pretrained = AutoModelForCausalLM.from_pretrained
    from transformers import cache_utils
    from transformers.models.gemma2 import modeling_gemma2
    original_hybrid_init = cache_utils.HybridCache.__init__
    def patched_hybrid_init(self, config, batch_size, max_cache_len, device, dtype=None, **kwargs):
        original_hybrid_init(self, config, batch_size, max_cache_len, device, dtype=torch.bfloat16, **kwargs)
    cache_utils.HybridCache.__init__ = patched_hybrid_init
    modeling_gemma2.HybridCache.__init__ = patched_hybrid_init
    
    def quantized_lora_from_pretrained(*args, **kwargs):
        # 1. Inject 4-bit Quantization Config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16
        )
        kwargs['quantization_config'] = bnb_config
        kwargs['device_map'] = 'auto'
        kwargs['torch_dtype'] = torch.bfloat16
        # 2. Load the base model using original HF function
        model = original_from_pretrained(*args, **kwargs)

        # 3. Prepare for k-bit training (freeze base weights, handle layer norms)
        model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant": False})

        # 4. Inject LoRA adapters
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.config.use_cache = False   
        model.generation_config.use_cache = False

        print("LoRA successfully applied to internal cell2sentence model")
        model.print_trainable_parameters()
        return model

    # Apply the patch immediately before CSModel initialization
    AutoModelForCausalLM.from_pretrained = quantized_lora_from_pretrained
    # --------------------------------------------

    # Load model (triggers the monkey patch)
    csmodel = cs.CSModel(
        model_name_or_path=model_path,
        save_dir=save_dir,
        save_name=save_name,
    )
    
    # Resolve output dir
    training_task = "cell_type_prediction"
    datetimestamp = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    output_dir = os.path.join(csmodel.save_dir, f"{datetimestamp}_finetune_{training_task}")
    os.makedirs(output_dir, exist_ok=True)
    train_args = TrainingArguments(**{**train_args.to_dict(), "output_dir": output_dir})

    # Fine-tune
    csmodel.fine_tune(
        csdata=csdata,
        task=training_task,
        train_args=train_args,
        loss_on_response_only=False,
        top_k_genes=top_k_genes,
        max_eval_samples=max_eval_samples,
    )
    AutoModelForCausalLM.from_pretrained = original_from_pretrained
    cache_utils.HybridCache.__init__ = original_hybrid_init
    modeling_gemma2.HybridCache.__init__ = original_hybrid_init

# =================================================
#    scGPT Finetuning
# =================================================

# Default training arguments
# -------------------------------------------
def train_scgpt_cell_classifier(
        adata,
        celltype_col,
        name,
        base_model,
        output_dir
):
    import shutil
    hyperparameter_defaults = dict(
        seed=0,
        dataset_name=name,
        do_train=True,
        load_model=base_model,
        mask_ratio=0.0,
        epochs=10,
        n_bins=51,
        MVC=False, # Masked value prediction for cell embedding
        ecs_thres=0.0, # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
        dab_weight=0.0,
        lr=1e-4,
        batch_size=32,
        layer_size=128,
        nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead=4,  # number of heads in nn.MultiheadAttention
        dropout=0.2,  # dropout probability
        schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
        save_eval_interval=5,
        fast_transformer=True,
        pre_norm=False,
        amp=True,  # Automatic Mixed Precision
        include_zero_gene = False,
        freeze = False, #freeze
        DSBN = False,  # Domain-spec batchnorm
    )

    def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
        masked_values_train = random_mask_value(
            tokenized_train["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        masked_values_valid = random_mask_value(
            tokenized_valid["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        print(
            f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
            f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
        )

        input_gene_ids_train, input_gene_ids_valid = (
            tokenized_train["genes"],
            tokenized_valid["genes"],
        )
        input_values_train, input_values_valid = masked_values_train, masked_values_valid
        target_values_train, target_values_valid = (
            tokenized_train["values"],
            tokenized_valid["values"],
        )

        tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
        tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

        tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
        tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

        if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
            train_sort_ids = np.argsort(train_batch_labels)
            input_gene_ids_train = input_gene_ids_train[train_sort_ids]
            input_values_train = input_values_train[train_sort_ids]
            target_values_train = target_values_train[train_sort_ids]
            tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
            tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

            valid_sort_ids = np.argsort(valid_batch_labels)
            input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
            input_values_valid = input_values_valid[valid_sort_ids]
            target_values_valid = target_values_valid[valid_sort_ids]
            tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
            tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

        train_data_pt = {
            "gene_ids": input_gene_ids_train,
            "values": input_values_train,
            "target_values": target_values_train,
            "batch_labels": tensor_batch_labels_train,
            "celltype_labels": tensor_celltype_labels_train,
        }
        valid_data_pt = {
            "gene_ids": input_gene_ids_valid,
            "values": input_values_valid,
            "target_values": target_values_valid,
            "batch_labels": tensor_batch_labels_valid,
            "celltype_labels": tensor_celltype_labels_valid,
        }

        return train_data_pt, valid_data_pt

    # dataset
    class SeqDataset(Dataset):
        def __init__(self, data: Dict[str, torch.Tensor]):
            self.data = data

        def __len__(self):
            return self.data["gene_ids"].shape[0]

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}

    # data_loader
    def prepare_dataloader(
        data_pt: Dict[str, torch.Tensor],
        batch_size: int,
        shuffle: bool = False,
        intra_domain_shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        if num_workers == 0:
            num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

        dataset = SeqDataset(data_pt)

        if per_seq_batch_sample:
            # find the indices of samples in each seq batch
            subsets = []
            batch_labels_array = data_pt["batch_labels"].numpy()
            for batch_label in np.unique(batch_labels_array):
                batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
                subsets.append(batch_indices)
            data_loader = DataLoader(
                dataset=dataset,
                batch_sampler=SubsetsBatchSampler(
                    subsets,
                    batch_size,
                    intra_subset_shuffle=intra_domain_shuffle,
                    inter_subset_shuffle=shuffle,
                    drop_last=drop_last,
                ),
                num_workers=num_workers,
                pin_memory=True,
            )
            return data_loader

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    def train(model: nn.Module, loader: DataLoader) -> None:
        """
        Train the model for one epoch.
        """
        model.train()
        (
            total_loss,
            total_mse,
            total_cls,
            total_cce,
            total_mvc,
            total_ecs,
            total_dab,
            total_adv_E,
            total_adv_D,
            total_zero_log_prob,
            total_mvc_zero_log_prob,
        ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        total_error = 0.0
        start_time = time.time()

        num_batches = len(loader)
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                    do_sample=do_sample_in_train,
                    #generative_training=False
                )

                masked_positions = input_values.eq(mask_value)  # the postions to predict
                loss = 0.0
                metrics_to_log = {}
                if MLM:
                    loss_mse = criterion(
                        output_dict["mlm_output"], target_values, masked_positions
                    )
                    loss = loss + loss_mse
                    metrics_to_log = {"train/mse": loss_mse.item()}
                if explicit_zero_prob:
                    loss_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mlm_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_zero_log_prob
                    metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                if CLS:
                    loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                    loss = loss + loss_cls
                    metrics_to_log.update({"train/cls": loss_cls.item()})

                    error_rate = 1 - (
                        (output_dict["cls_output"].argmax(1) == celltype_labels)
                        .sum()
                        .item()
                    ) / celltype_labels.size(0)
                if CCE:
                    loss_cce = 10 * output_dict["loss_cce"]
                    loss = loss + loss_cce
                    metrics_to_log.update({"train/cce": loss_cce.item()})
                if MVC:
                    loss_mvc = criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )
                    loss = loss + loss_mvc
                    metrics_to_log.update({"train/mvc": loss_mvc.item()})
                if MVC and explicit_zero_prob:
                    loss_mvc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_mvc_zero_log_prob
                    metrics_to_log.update({"train/mvc_nzlp": loss_mvc_zero_log_prob.item()})
                if ECS:
                    loss_ecs = 10 * output_dict["loss_ecs"]
                    loss = loss + loss_ecs
                    metrics_to_log.update({"train/ecs": loss_ecs.item()})
                if DAB:
                    # try weighting and separate optimizer
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                    loss = loss + dab_weight * loss_dab
                    metrics_to_log.update({"train/dab": loss_dab.item()})

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()

            if ADV:
                # rerun the model for adversarial training
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                    do_sample=do_sample_in_train,
                    #generative_training=False
                )

                # TRAINING DISCRIMINATOR
                loss_adv_D = criterion_adv(
                    discriminator(output_dict["cell_emb"].detach()), batch_labels
                )
                if epoch > adv_D_delay_epochs:
                    discriminator.zero_grad()
                    loss_adv_D.backward()
                    optimizer_D.step()

                # TRAINING ENCODER
                loss_adv_E = -criterion_adv(
                    discriminator(output_dict["cell_emb"]), batch_labels
                )
                # NOTE: the loss is negative here because we want to maximize
                # the cross_entropy_loss, in other words, disguise against the discriminator
                if epoch > adv_E_delay_epochs:
                    model.zero_grad()
                    discriminator.zero_grad()
                    loss_adv_E.backward()
                    optimizer_E.step()

            wandb.log(metrics_to_log)

            total_loss += loss.item()
            total_mse += loss_mse.item() if MLM else 0.0
            total_cls += loss_cls.item() if CLS else 0.0
            total_cce += loss_cce.item() if CCE else 0.0
            total_mvc += loss_mvc.item() if MVC else 0.0
            total_ecs += loss_ecs.item() if ECS else 0.0
            total_dab += loss_dab.item() if DAB else 0.0
            total_adv_E += loss_adv_E.item() if ADV else 0.0
            total_adv_D += loss_adv_D.item() if ADV else 0.0
            total_zero_log_prob += loss_zero_log_prob.item() if explicit_zero_prob else 0.0
            total_mvc_zero_log_prob += (
                loss_mvc_zero_log_prob.item() if MVC and explicit_zero_prob else 0.0
            )
            total_error += error_rate
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                cur_cls = total_cls / log_interval if CLS else 0.0
                cur_cce = total_cce / log_interval if CCE else 0.0
                cur_mvc = total_mvc / log_interval if MVC else 0.0
                cur_ecs = total_ecs / log_interval if ECS else 0.0
                cur_dab = total_dab / log_interval if DAB else 0.0
                cur_adv_E = total_adv_E / log_interval if ADV else 0.0
                cur_adv_D = total_adv_D / log_interval if ADV else 0.0
                cur_zero_log_prob = (
                    total_zero_log_prob / log_interval if explicit_zero_prob else 0.0
                )
                cur_mvc_zero_log_prob = (
                    total_mvc_zero_log_prob / log_interval
                    if MVC and explicit_zero_prob
                    else 0.0
                )
                cur_error = total_error / log_interval
                # ppl = math.exp(cur_loss)
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | "
                    + (f"mse {cur_mse:5.2f} | mre {cur_error:5.2f} |" if MLM else "")
                    + (f"cls {cur_cls:5.2f} | " if CLS else "")
                    + (f"err {cur_error:5.2f} | " if CLS else "")
                    + (f"cce {cur_cce:5.2f} |" if CCE else "")
                    + (f"mvc {cur_mvc:5.2f} |" if MVC else "")
                    + (f"ecs {cur_ecs:5.2f} |" if ECS else "")
                    + (f"dab {cur_dab:5.2f} |" if DAB else "")
                    + (f"adv_E {cur_adv_E:5.2f} |" if ADV else "")
                    + (f"adv_D {cur_adv_D:5.2f} |" if ADV else "")
                    + (f"nzlp {cur_zero_log_prob:5.2f} |" if explicit_zero_prob else "")
                    + (
                        f"mvc_nzlp {cur_mvc_zero_log_prob:5.2f} |"
                        if MVC and explicit_zero_prob
                        else ""
                    )
                )
                total_loss = 0
                total_mse = 0
                total_cls = 0
                total_cce = 0
                total_mvc = 0
                total_ecs = 0
                total_dab = 0
                total_adv_E = 0
                total_adv_D = 0
                total_zero_log_prob = 0
                total_mvc_zero_log_prob = 0
                total_error = 0
                start_time = time.time()


    def define_wandb_metrcis():
        wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
        wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
        wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
        wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
        wandb.define_metric("test/avg_bio", summary="max")


    def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
        """
        Evaluate the model on the evaluation data.
        """
        model.eval()
        total_loss = 0.0
        total_error = 0.0
        total_dab = 0.0
        total_num = 0
        predictions = []
        with torch.no_grad():
            for batch_data in loader:
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                target_values = batch_data["target_values"].to(device)
                batch_labels = batch_data["batch_labels"].to(device)
                celltype_labels = batch_data["celltype_labels"].to(device)

                src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
                with torch.cuda.amp.autocast(enabled=config.amp):
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                        CLS=CLS,  # evaluation does not need CLS or CCE
                        CCE=False,
                        MVC=False,
                        ECS=False,
                        do_sample=do_sample_in_train,
                        #generative_training = False,
                    )
                    output_values = output_dict["cls_output"]
                    loss = criterion_cls(output_values, celltype_labels)

                    if DAB:
                        loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

                total_loss += loss.item() * len(input_gene_ids)
                accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
                total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
                total_num += len(input_gene_ids)
                preds = output_values.argmax(1).cpu().numpy()
                predictions.append(preds)

        wandb.log(
            {
                "valid/mse": total_loss / total_num,
                "valid/err": total_error / total_num,
                "valid/dab": total_dab / total_num,
                "valid/sum_mse_dab": (total_loss + dab_weight * total_dab) / total_num,
                "epoch": epoch,
            },
        )

        if return_raw:
            return np.concatenate(predictions, axis=0)

        return total_loss / total_num, total_error / total_num
    
    # %% inference
    def test(model: nn.Module, adata: DataLoader) -> float:
        all_counts = (
            adata.layers[input_layer_key].A
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )

        celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
        celltypes_labels = np.array(celltypes_labels)

        batch_ids = adata.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)

        tokenized_test = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=include_zero_gene,
        )

        input_values_test = random_mask_value(
            tokenized_test["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )

        test_data_pt = {
            "gene_ids": tokenized_test["genes"],
            "values": input_values_test,
            "target_values": tokenized_test["values"],
            "batch_labels": torch.from_numpy(batch_ids).long(),
            "celltype_labels": torch.from_numpy(celltypes_labels).long(),
        }

        test_loader = DataLoader(
            dataset=SeqDataset(test_data_pt),
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
            pin_memory=True,
        )

        model.eval()
        predictions = evaluate(
            model,
            loader=test_loader,
            return_raw=True,
        )

        # compute accuracy, precision, recall, f1
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(celltypes_labels, predictions)
        precision = precision_score(celltypes_labels, predictions, average="macro")
        recall = recall_score(celltypes_labels, predictions, average="macro")
        macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

        logger.info(
            f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
            f"Macro F1: {macro_f1:.3f}"
        )

        results = {
            "test/accuracy": accuracy,
            "test/precision": precision,
            "test/recall": recall,
            "test/macro_f1": macro_f1,
        }
        return predictions, celltypes_labels, results

    run = wandb.init(
        config=hyperparameter_defaults,
        project="scGPT",
        reinit=True,
        mode="offline",
        settings=wandb.Settings(start_method="fork"),
    )
    config = wandb.config
    print(config)
    set_seed(config.seed)

    # settings for input and preprocessing
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    mask_ratio = config.mask_ratio
    mask_value = "auto"  # for masked values, now it should always be auto

    include_zero_gene = config.include_zero_gene  # if True, include zero genes among hvgs in the training
    max_seq_len = 3001
    n_bins = config.n_bins
    
    # input/output representation
    input_style = "binned"  # "normed_raw", "log1p", or "binned"
    output_style = "binned"  # "normed_raw", "log1p", or "binned"

    # settings for training
    MLM = False  # whether to use masked language modeling, currently it is always on.
    CLS = True  # celltype classification objective
    ADV = False  # Adversarial training for batch correction
    CCE = False  # Contrastive cell embedding objective
    MVC = config.MVC  # Masked value prediction for cell embedding
    ECS = config.ecs_thres > 0  # Elastic cell similarity objective
    DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
    INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
    input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
    cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
    adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
    adv_D_delay_epochs = 0
    mvc_decoder_style = "inner product"
    ecs_threshold = config.ecs_thres
    dab_weight = config.dab_weight

    explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
    do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

    per_seq_batch_sample = False
    # settings for optimizer
    lr = config.lr  # TODO: test learning rate ratio between two tasks
    lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
    batch_size = config.batch_size
    eval_batch_size = config.batch_size
    epochs = config.epochs
    schedule_interval = 1

    # settings for the model
    fast_transformer = config.fast_transformer
    fast_transformer_backend = "flash"  # "linear" or "flash"
    embsize = config.layer_size  # embedding dimension
    d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
    nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
    nhead = config.nhead  # number of heads in nn.MultiheadAttention
    dropout = config.dropout  # dropout probability

    # logging
    log_interval = 100  # iterations
    save_eval_interval = config.save_eval_interval  # epochs
    do_eval_scib_metrics = True

    # %% validate settings
    assert input_style in ["normed_raw", "log1p", "binned"]
    assert output_style in ["normed_raw", "log1p", "binned"]
    assert input_emb_style in ["category", "continuous", "scaling"]
    if input_style == "binned":
        if input_emb_style == "scaling":
            raise ValueError("input_emb_style `scaling` is not supported for binned input.")
    elif input_style == "log1p" or input_style == "normed_raw":
        if input_emb_style == "category":
            raise ValueError(
                "input_emb_style `category` is not supported for log1p or normed_raw input."
        )   

    if input_emb_style == "category":
        mask_value = n_bins + 1
        pad_value = n_bins  # for padding gene expr values
        n_input_bins = n_bins + 2
    else:
        mask_value = -1
        pad_value = -2
        n_input_bins = n_bins

    if ADV and DAB:
        raise ValueError("ADV and DAB cannot be both True.")
    DAB_separate_optim = True if DAB > 1 else False

    dataset_name = config.dataset_name
    save_dir = Path(f"{output_dir}/save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")

    adata.obs["celltype"] = adata.obs[celltype_col].astype("category")
    adata.obs["celltype"] = adata.obs["celltype"].cat.remove_unused_categories()
    adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
    data_is_raw = False
    filter_gene_by_counts = False
    metadata = adata.obs
    metadata["cell_id"] = metadata.index
    X = metadata[['cell_id']]
    y = metadata['celltype']
    train_data, eval_data = train_test_split(metadata,
                                             test_size=0.2,
                                             random_state=42,
                                             stratify=metadata["celltype"])
    adata.obs["batch_id"] = ["1" if x in eval_data.index else "0" for x in adata.obs.index]
    adata.obs["str_batch"] = ["1" if x in eval_data.index else "0" for x in adata.obs.index]

    # make the batch category column
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    celltypes = adata.obs["celltype"].unique()
    num_types = len(np.unique(celltype_id_labels))
    id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
    adata.obs["celltype_id"] = celltype_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()

    if config.load_model is not None:
        model_dir = Path(config.load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)
        shutil.copy(vocab_file, save_dir / "vocab.json")
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

    # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]

    # set up the preprocessor, use the args to config the workflow
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=filter_gene_by_counts,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=data_is_raw,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )

    adata_test = adata[adata.obs["str_batch"] == "1"]
    adata = adata[adata.obs["str_batch"] == "0"]

    preprocessor(adata, batch_key=None)
    preprocessor(adata_test, batch_key=None)

    input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[input_style]
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()

    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    num_batch_types = len(set(batch_ids))
    batch_ids = np.array(batch_ids)

    (
        train_data,
        valid_data,
        train_celltype_labels,
        valid_celltype_labels,
        train_batch_labels,
        valid_batch_labels,
    ) = train_test_split(
        all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
    )

    if config.load_model is None:
        vocab = Vocab(
            VocabPybind(genes + special_tokens, None)
        )  # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=include_zero_gene,
    )
    logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=3,
        n_cls=num_types if CLS else 1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=MVC,
        do_dab=DAB,
        use_batch_labels=INPUT_BATCH_LABELS,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=config.DSBN,
        input_emb_style=input_emb_style,
        n_input_bins=n_input_bins,
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        ecs_threshold=ecs_threshold,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=fast_transformer,
        fast_transformer_backend=fast_transformer_backend,
        pre_norm=config.pre_norm,
    )

    if config.load_model is not None:
        try:
            model.load_state_dict(torch.load(model_file))
            logger.info(f"Loading all model params from {model_file}")
        except:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    # Freeze all pre-decoder weights
    for name, para in model.named_parameters():
        print("-"*20)
        print(f"name: {name}")
        if config.freeze and "encoder" in name and "transformer_encoder" not in name:
        # if config.freeze and "encoder" in name:
            print(f"freezing weights for: {name}")
            para.requires_grad = False

    post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
    logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")
    wandb.log(
            {
                "info/pre_freeze_param_count": pre_freeze_param_count,
                "info/post_freeze_param_count": post_freeze_param_count,
            },
    )

    model.to(device)
    wandb.watch(model)

    if ADV:
        discriminator = AdversarialDiscriminator(
            d_model=embsize,
            n_cls=num_batch_types,
        ).to(device)

    criterion = masked_mse_loss
    criterion_cls = nn.CrossEntropyLoss()
    criterion_dab = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, schedule_interval, gamma=config.schedule_ratio
    )
    if DAB_separate_optim:
        optimizer_dab = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler_dab = torch.optim.lr_scheduler.StepLR(
            optimizer_dab, schedule_interval, gamma=config.schedule_ratio
        )
    if ADV:
        criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
        optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV)
        scheduler_E = torch.optim.lr_scheduler.StepLR(
            optimizer_E, schedule_interval, gamma=config.schedule_ratio
        )
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV)
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            optimizer_D, schedule_interval, gamma=config.schedule_ratio
        )   

    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
    best_val_loss = float("inf")
    best_avg_bio = 0.0
    best_model = None
    define_wandb_metrcis()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=batch_size,
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
        )
        valid_loader = prepare_dataloader(
            valid_data_pt,
            batch_size=eval_batch_size,
            shuffle=False,
            intra_domain_shuffle=False,
            drop_last=False,
        )

        if config.do_train:
            train(
                model,
                loader=train_loader,
            )
        val_loss, val_err = evaluate(
            model,
            loader=valid_loader,
        )
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
        )
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model with score {best_val_loss:5.4f}")
        
        scheduler.step()
        if DAB_separate_optim:
            scheduler_dab.step()
        if ADV:
            scheduler_D.step()
            scheduler_E.step()

    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    # copy args file over from base model
    import shutil
    shutil.copy(Path(base_model) / "args.json", save_dir / "args.json")
    predictions, labels, results = test(best_model, adata_test)
    from sklearn.metrics import confusion_matrix
    celltypes = list(celltypes)
    for i in set([id2type[p] for p in predictions]):
        if i not in celltypes:
            celltypes.remove(i)
    cm = confusion_matrix(labels, predictions)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm = pd.DataFrame(cm, index=celltypes[:cm.shape[0]], columns=celltypes[:cm.shape[
    1]])
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
    plt.savefig(save_dir / "confusion_matrix.png", dpi=300)

    results["test/confusion_matrix"] = wandb.Image(
        str(save_dir / "confusion_matrix.png"),
        caption=f"confusion matrix",
    )
