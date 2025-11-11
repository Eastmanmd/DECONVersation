#!/bin/bash
#SBATCH --job-name=train_rf_model_pca
#SBATCH --output=logs/train_rf_model_pca.log   
#SBATCH --error=logs/train_rf_model_pca.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=cpu  

# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/lineage/train_random_forest_regr_deconv.py \
--train_latent /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/train/pca_pseudobulk_train_set.csv \
--train_prop /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/train/cell_prop.csv \
--test_latent /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/test/pca_pseudobulk_test_set.csv \
--out_val_pred /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/metrics/pca_output_val_predictions.csv \
--out_test_pred /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/metrics/pca_output_test_predictions.csv \
--out_model /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/models/pca_best_model.pkl