#!/bin/bash
#SBATCH --job-name=get_gf_embed_sig_mat
#SBATCH --output=logs/get_gf_embed_sig_mat.log   
#SBATCH --error=logs/get_gf_embed_sig_mat.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=gpu  
#SBATCH --gres=gpu 


# Run script
/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/Python/get_geneformer_embeddings.py \
--input_csv /gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvBench/haoSub_signature_matrix_broad.csv \
--token_output_dir /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_hoek/train/tokenize_sig_mat \
--geneformer_model_path /gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvBench/geneformer/complete/finetuned/260202_geneformer_cellClassifier_cell_annot_subset/ksplit1 \
--token_output_name temp_tokens \
--embedding_output_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/hoek/train/gf_embeddings_signature_matrix_for_nnls_complete.csv