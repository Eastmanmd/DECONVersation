#!/bin/bash
#SBATCH --job-name=run_nnls_gf_bulk
#SBATCH --output=logs/run_nnls_gf_bulk.log   
#SBATCH --error=logs/run_nnls_gf_bulk.err   
#SBATCH --time=72:00:00
#SBATCH --mem=8G                     
#SBATCH --partition=cpu  

# Run script
/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/Python/run_nnls.py \
--pseudobulk /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/pseudo_bulk/gf_czi_pseudobulk_embed_zeroshot_cls_emb_0.csv \
--signature /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/pseudo_bulk/gf_czi_sig_mat_pseudobulk_embed_zeroshot_cls_emb_0.csv \
--output /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/pseudo_bulk/gf_pb_nnls_cell_proportions_finetuned_cls_emb_0.csv