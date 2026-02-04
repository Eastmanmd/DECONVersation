#!/bin/bash
#SBATCH --job-name=run_nnls_counts
#SBATCH --output=logs/run_nnls_counts.log   
#SBATCH --error=logs/run_nnls_counts.err   
#SBATCH --time=72:00:00
#SBATCH --mem=8G                     
#SBATCH --partition=cpu  

# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/lineage/run_nnls.py \
--pseudobulk /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/test/pseudobulk_raw.csv \
--signature /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/train/signature_matrix_for_nnls.csv \
--output /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/test/nnls_cell_proportions_counts.csv