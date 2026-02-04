#!/bin/bash
#SBATCH --job-name=run_nnls_counts
#SBATCH --output=logs/run_nnls_counts.log   
#SBATCH --error=logs/run_nnls_counts.err   
#SBATCH --time=72:00:00
#SBATCH --mem=8G                     
#SBATCH --partition=cpu  

# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/Python/run_nnls.py \
--pseudobulk /gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/GSE220608/GSE220605_DataMatrix_Mixture_CellLines_Bulk_RNA-seq_gene_ids.csv \
--signature /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/train/signature_matrix_for_nnls.csv \
--output /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/metrics_bulk_nnls/counts_output_test_predictions.csv