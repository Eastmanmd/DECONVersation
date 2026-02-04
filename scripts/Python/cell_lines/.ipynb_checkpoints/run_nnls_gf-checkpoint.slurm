#!/bin/bash
#SBATCH --job-name=run_nnls_gf
#SBATCH --output=logs/run_nnls_gf.log   
#SBATCH --error=logs/run_nnls_gf.err   
#SBATCH --time=72:00:00
#SBATCH --mem=8G                     
#SBATCH --partition=cpu  

# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/Python/run_nnls.py \
--pseudobulk /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/temp/gf_embed_mixture_cell_lines.csv \
--signature /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/temp/gf_embed_pure_cell_lines.csv \
--output /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/metrics_nnls/gf_nnls_cell_proportions_bulk_cell_lines.csv