#!/bin/bash
#SBATCH --job-name=run_nnls_c2s_bulk
#SBATCH --output=logs/run_nnls_c2s_bulk.log   
#SBATCH --error=logs/run_nnls_c2s_bulk.err   
#SBATCH --time=72:00:00
#SBATCH --mem=8G                     
#SBATCH --partition=cpu  

# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/Python/run_nnls.py \
--pseudobulk /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/test/c2s_embeddings_cell_lines_bulk_refinetune.csv \
--signature /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/train/c2s_embeddings_refinetune_signature_matrix_for_nnls.csv \
--output /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/metrics_bulk_nnls/c2s_refinetune_output_test_predictions.csv