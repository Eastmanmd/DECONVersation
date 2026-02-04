#!/bin/bash
#SBATCH --job-name=get_scVI_embed_sig_mat
#SBATCH --output=logs/get_scVI_embed_sig_mat.log   
#SBATCH --error=logs/get_scVI_embed_sig_mat.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=cpu

# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/lineage/get_scVI_embeddings.py \
--input_csv /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/train/signature_matrix_for_nnls.csv \
--scvi_model_path /gpfs/commons/groups/compbio/projects/ao_projects/scvi-human-2024-07-01 \
--embedding_output_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/train/scVI_sig_mat_embeddings.csv