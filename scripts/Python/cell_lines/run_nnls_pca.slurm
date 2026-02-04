#!/bin/bash
#SBATCH --job-name=run_nnls_pca
#SBATCH --output=logs/run_nnls_pca.log   
#SBATCH --error=logs/run_nnls_pca.err   
#SBATCH --time=72:00:00
#SBATCH --mem=8G                     
#SBATCH --partition=cpu  

# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/lineage/run_nnls.py \
--pseudobulk /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/test/pca_pseudobulk.csv \
--signature /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/train/pca_signature_matrix_for_nnls.csv \
--output /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/metrics_nnls/pca_nnls_cell_proportions.csv