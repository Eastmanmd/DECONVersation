#!/bin/bash
#SBATCH --job-name=train_rf_model_c2s_bulk
#SBATCH --output=logs/train_rf_model_c2s_bulk.log   
#SBATCH --error=logs/train_rf_model_c2s_bulk.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=cpu  

# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/Python/train_random_forest_regr_deconv.py \
--train_latent /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/train/c2s_embeddings_pseudobulk.csv \
--train_prop /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/train/cell_proportions.csv \
--test_latent /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/test/c2s_embeddings_cell_lines_bulk.csv \
--out_val_pred /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/metrics_bulk/c2s_output_val_predictions.csv \
--out_test_pred /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/metrics_bulk/c2s_output_test_predictions.csv \
--out_model /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/models/c2s_best_model_bulk.pkl \
--log_file /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/Python/cell_lines/log_files/c2s_random_forest_bulk.log
