#!/bin/bash
#SBATCH --job-name=get_gf_embed
#SBATCH --output=logs/get_gf_embed.log   
#SBATCH --error=logs/get_gf_embed.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=gpu  
#SBATCH --gres=gpu 


# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/lineage/get_geneformer_embeddings.py \
--input_csv /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/train/signature_matrix_for_nnls.csv \
--token_output_dir /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/train/tokenize_sig_mat \
--token_output_name temp_tokens \
--embedding_output_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/train/gf_embeddings_signature_matrix_for_nnls.csv