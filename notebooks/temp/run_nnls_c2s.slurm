#!/bin/bash
#SBATCH --job-name=run_nnls_c2s
#SBATCH --output=logs/run_nnls_c2s.log   
#SBATCH --error=logs/run_nnls_c2s.err   
#SBATCH --time=72:00:00
#SBATCH --mem=8G                     
#SBATCH --partition=cpu  

# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/lineage/run_nnls.py \
--pseudobulk /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/test/c2s_pseudobulk_test_set_genename.csv \
--signature /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/train/c2s_signature_matrix_for_nnls_genename.csv \
--output /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/test/nnls_cell_proportions_c2s.csv