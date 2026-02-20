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
import torch

adata_path = "/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/single_cell_refs/CZI04N_post_menopausal_sequence_duplicates.h5ad"
c2s_save_dir = "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/czi_sub/c2s"
c2s_save_name = "CZI04N"
cell_type_prediction_model_path = "/gpfs/commons/groups/compbio/projects/rf_projects/rf_packages/C2S-Pythia-410m-cell-type-prediction"
save_dir = "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvbench/c2s/410mre"
save_name = "cell_embedding_prediction_C2S-Scale-Pythia-410m_refinetune"
adata = anndata.read_h5ad(adata_path)
adata.obs["organism"]="Homo sapiens"
adata.obs["cell_type"]=adata.obs["heca_lineage"]
adata.obs["tissue"]="endom"
adata.obs["sex"]="F"
adata.obs["batch_condition"]=adata.obs["genotype_call"]
adata_obs_cols_to_keep = ["organism","cell_type","tissue", "sex", "batch_condition"]

arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
    adata=adata,
    random_state=SEED,
    sentence_delimiter=' ',
    label_col_names=adata_obs_cols_to_keep
)
csdata = cs.CSData.csdata_from_arrow(
    arrow_dataset=arrow_ds, 
    vocabulary=vocabulary,
    save_dir=c2s_save_dir,
    save_name=c2s_save_name,
    dataset_backend="arrow"
)
csmodel = cs.CSModel(
    model_name_or_path=cell_type_prediction_model_path,
    save_dir=save_dir,
    save_name=save_name
)
training_task = "cell_type_prediction"
datetimestamp = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
output_dir = os.path.join(csmodel.save_dir, datetimestamp + f"_finetune_{training_task}")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
print(output_dir)
del arrow_ds
del adata
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
train_args = TrainingArguments(
    bf16=True,
    fp16=False,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    learning_rate=1e-5,
    load_best_model_at_end=True,
    logging_steps=50,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    num_train_epochs=10, 
    eval_steps=50,
    evaluation_strategy="steps",
    save_steps=100,
    save_strategy="steps",
    save_total_limit=3,
    warmup_ratio=0.05,
    output_dir=output_dir,
    torch_empty_cache_steps=1
)
csmodel.fine_tune(
    csdata=csdata,
    task=training_task,
    train_args=train_args,
    loss_on_response_only=False,
    top_k_genes=200,
    max_eval_samples=500,
)
