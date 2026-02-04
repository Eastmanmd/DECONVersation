#!/bin/bash
#SBATCH --job-name=get_gf_embed_sig_mat
#SBATCH --output=logs/get_gf_embed_sig_mat.log   
#SBATCH --error=logs/get_gf_embed_sig_mat.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=gpu  
#SBATCH --gres=gpu 


# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/lineage/get_geneformer_embeddings.py \
--input_csv /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/train/signature_matrix_for_nnls.csv \
--token_output_dir /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/train/tokenize_sig_mat \
--token_output_name sig_mat_tokenize \
--embedding_output_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/train/gf_sig_mat_embeddings.csv