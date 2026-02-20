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

adata_path = "/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/single_cell_refs/CZI04N_post_menopausal_sequence_duplicates.h5ad"
c2s_save_dir = "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/czi_sub/c2s/2b"
c2s_save_name = "CZI04N"
cell_type_prediction_model_path = "/gpfs/commons/groups/compbio/projects/rf_projects/rf_packages/C2S-Scale-Gemma-2-2B"
save_dir = "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvbench/c2s/2b"
save_name = "cell_embedding_prediction_C2S-Scale-Gemma-2-2B_finetune"
adata = anndata.read_h5ad(adata_path)
adata.obs["organism"]="Homo sapiens"
adata.obs["cell_type"]=adata.obs["heca_lineage"]
adata.obs["tissue"]="endom"
adata.obs["sex"]="F"
adata.obs["batch_condition"]=adata.obs["genotype_call"]
adata_obs_cols_to_keep = ["organism","cell_type","tissue", "sex", "batch_condition"]
arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
    adata=adata, 
    random_state=42, 
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

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from cell2sentence import CSModel

# Save original
original_from_pretrained = AutoModelForCausalLM.from_pretrained

def quantized_lora_from_pretrained(*args, **kwargs):
    # Add quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    kwargs['quantization_config'] = bnb_config
    kwargs['device_map'] = 'auto'

    # Load quantized model
    model = original_from_pretrained(*args, **kwargs)

    # Add LoRA adapters
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    print(f"LoRA applied! Trainable params:")
    model.print_trainable_parameters()

    return model

# Monkey patch
AutoModelForCausalLM.from_pretrained = quantized_lora_from_pretrained

# Now use CSModel normally
csmodel = cs.CSModel(
    model_name_or_path=cell_type_prediction_model_path,
    save_dir=save_dir,
    save_name=save_name
)
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
# Continue with fine_tune() as before

train_args = TrainingArguments(
    bf16=True,
    fp16=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    load_best_model_at_end=True,
    logging_steps=50,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    num_train_epochs=5, 
    eval_steps=50,
    evaluation_strategy="steps",
    save_steps=100,
    save_strategy="steps",
    save_total_limit=2,
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
    max_eval_samples=200,
)
