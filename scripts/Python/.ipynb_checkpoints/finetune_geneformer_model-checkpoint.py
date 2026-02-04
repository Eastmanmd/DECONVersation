#!/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python

# Goal: Finetune geneformer for cell classification. The goal is to use this model for downstream analyses specifically geneformer perturbations.
# To avoid model bias towards over represented cells, this script will contain code to randomly undersampling some oversampled cell types

# Import libraries
import pickle
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import scipy.sparse as sp 
import matplotlib.pyplot as plt
import anndata
from sklearn.model_selection import train_test_split


# For PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# For umap
import umap

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from geneformer import Classifier

# Load adata
path = "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvBench/geneformer/complete/haoSub.h5ad"

# Read data
adata = sc.read_h5ad(path)

# Get current date (geneformer creates model directory using date time information)
from datetime import datetime
now = datetime.now()
date_string = now.strftime("%y%m%d")

# Set output directory
output_dir = f"/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvBench/geneformer/complete/finetuned/"

# Limit to only select cell types
# type = broad_cell_type
filter_data_dict = {"type": ["T cells",
                             "Monocytes",  
                             "NK cells",                   
                             "Platelet",
                             "mDC",          
                             "B cells",                
                             "pDC"]}
    
# Set training arguments
training_args = {
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

# Set up classifier
cc = Classifier(classifier="cell",
                cell_state_dict = {"state_key": "type", "states": "all"},
                filter_data=filter_data_dict,
                training_args=training_args,
                max_ncells=None,
                freeze_layers = 2,
                num_crossval_splits = 1,
                forward_batch_size=10,
                nproc=8)


# Access metadata
metadata = adata.obs
metadata["cell_id"] = metadata.index

X = metadata[['cell_id']]
y = metadata['type']


# Define number to sample per class
n_per_class = 2000

# #Sample set cells per 'type' category (or fewer if not enough)
metadata_balanced = (
    metadata
    .groupby('type', group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), n_per_class), random_state=42))
)


# Find the cell IDs in the resampled metadata
resampled_cell_ids = metadata_balanced.index

# Subset the original metadata to get the rows not included in metadata_balanced
metadata_not_in_balanced = metadata[~metadata['cell_id'].isin(resampled_cell_ids)]

# Display the result
print("Dim of samples INCLUDED in training data")
print(metadata_balanced.shape)
print("\n")

print("Dim of samples NOT INCLUDED in training data")
print(metadata_not_in_balanced.shape)
print("\n")

# Reduce size of test data set for faster evaluation
test_data = (
    metadata_not_in_balanced
    .groupby('type', group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), 800), random_state=42))
)


# Split data into train (80%) and temporary (20%)
train_data, eval_data = train_test_split(metadata_balanced, 
                                         test_size=0.2, 
                                         random_state=42, 
                                         stratify=metadata_balanced["type"])

train_test_id_split_dict = {"attr_key": "cell_id",
                            "train": list(train_data.index)+list(eval_data.index),
                            "test": test_data.index}

# Prepare data for training
cc.prepare_data(input_data_file="/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvBench/geneformer/complete/tokenize/token_hoek.dataset/",
                output_directory=output_dir,
                output_prefix="cell_annot_subset",
                split_id_dict=train_test_id_split_dict)

# Separate train data from eval data
train_valid_id_split_dict = {"attr_key": "cell_id",
                            "train": list(train_data.index),
                            "eval": list(eval_data.index)}

print("training size")
print(len(train_valid_id_split_dict["train"]))
print("\n")


# Train cell classifier
print("training")
print("\n")

# Using updated model
all_metrics = cc.validate(model_directory="ctheodoris/Geneformer",
                          prepared_input_data_file="/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvBench/geneformer/complete/finetuned/cell_annot_subset_labeled_train.dataset/",
                          id_class_dict_file="/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvBench/geneformer/complete/finetuned/cell_annot_subset_id_class_dict.pkl",
                          output_directory=output_dir,
                          output_prefix="cell_annot_subset",
                          split_id_dict=train_valid_id_split_dict)

print("evaluating")
print("\n")

# Evaluate finetuned model on test dataset 
all_metrics_test = cc.evaluate_saved_model(
        model_directory=f'/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvBench/geneformer/complete/finetuned/{date_string}_geneformer_cellClassifier_cell_annot_subset/ksplit1/',
        id_class_dict_file="/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvBench/geneformer/complete/finetuned/cell_annot_subset_id_class_dict.pkl",
        test_data_file="/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvBench/geneformer/complete/finetuned/cell_annot_subset_labeled_test.dataset/",
        output_directory=output_dir,
        output_prefix="cell_annot_subset"
    )

print("done")
print("\n")



