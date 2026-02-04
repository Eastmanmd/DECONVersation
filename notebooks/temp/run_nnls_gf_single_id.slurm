#!/bin/bash
#SBATCH --job-name=run_nnls_gf_single
#SBATCH --output=logs/run_nnls_gf_single.log   
#SBATCH --error=logs/run_nnls_gf_single.err   
#SBATCH --time=72:00:00
#SBATCH --mem=8G                     
#SBATCH --partition=cpu  

# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/celltype/run_nnls.py \
--pseudobulk /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/single_samp/test/gf_zeroshot_pseudobulk_test_set.csv \
--signature /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/train/sig_mat_embeddings_temp.csv \
--output /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/test/nnls_cell_proportions_gf_single_samp.csv