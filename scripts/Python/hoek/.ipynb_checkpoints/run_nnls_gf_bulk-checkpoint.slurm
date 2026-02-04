#!/bin/bash
#SBATCH --job-name=run_nnls_gf_bulk
#SBATCH --output=logs/run_nnls_gf_bulk.log   
#SBATCH --error=logs/run_nnls_gf_bulk.err   
#SBATCH --time=72:00:00
#SBATCH --mem=8G                     
#SBATCH --partition=cpu  

# Run script
/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/Python/run_nnls.py \
--pseudobulk /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/hoek/train/gf_pseudobulk_embed_complete.csv \
--signature /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/hoek/train/gf_embeddings_signature_matrix_for_nnls_complete.csv \
--output /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_hoek/gf_nnls_cell_proportions.csv