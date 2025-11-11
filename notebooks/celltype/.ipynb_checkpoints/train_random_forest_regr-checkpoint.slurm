#!/bin/bash
#SBATCH --job-name=train_rf_model
#SBATCH --output=logs/train_rf_model.log   
#SBATCH --error=logs/train_rf_model.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=cpu  

# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/celltype/train_random_forest_regr_deconv.py \
--train_latent /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/train/gf_zeroshot_pseudobulk_train_set.csv \
--train_prop /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/train/cell_prop.csv \
--test_latent /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/single_samp/test/gf_zeroshot_pseudobulk_test_set.csv \
--out_val_pred /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/metrics/single_samp_output_val_predictions.csv \
--out_test_pred /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/metrics/single_samp_output_test_predictions.csv \
--out_model /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/models/single_samp_best_model.pkl