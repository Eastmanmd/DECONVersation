# ===============================
# Standard Libraries
# ===============================
import os
import pickle
import warnings
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ===============================
# For LoRA fine-tuning 
# ===============================
import torch
from datasets import load_from_disk
from transformers import BertForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType


# ===============================
# Geneformer
# ===============================
try:
    from geneformer import Classifier, TranscriptomeTokenizer
    from geneformer import DataCollatorForCellClassification
    print("geneformer successfully imported.")
    
except ImportError:
    print("geneformer is not installed. Skipping related functions.")


# Default training arguments
# -------------------------------------------------------------------
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

# =====================================
#    Full Geneformer Finetuning
# =====================================
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

    