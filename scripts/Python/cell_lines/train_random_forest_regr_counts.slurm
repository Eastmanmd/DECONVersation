#!/bin/bash
#SBATCH --job-name=train_rf_model_counts
#SBATCH --output=logs/train_rf_model_counts.log   
#SBATCH --error=logs/train_rf_model_counts.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=cpu  

# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/lineage/train_random_forest_regr_deconv.py \
--train_latent /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/train/pseudobulk_counts.csv \
--train_prop /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/train/cell_proportions.csv \
--test_latent /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/test/pseudobulk_counts.csv \
--out_val_pred /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/metrics/counts_output_val_predictions.csv \
--out_test_pred /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/metrics/counts_output_test_predictions.csv \
--out_model /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/models/counts_best_model.pkl \
--log_file /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/cell_lines/scripts/log_files/counts_random_forest.log