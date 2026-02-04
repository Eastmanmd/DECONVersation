#!/bin/bash
#SBATCH --job-name=run_nnls_c2s_27B
#SBATCH --output=logs/run_nnls_c2s_27B.log   
#SBATCH --error=logs/run_nnls_c2s_27B.err   
#SBATCH --time=72:00:00
#SBATCH --mem=8G                     
#SBATCH --partition=cpu  

# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/lineage/run_nnls.py \
--pseudobulk /gpfs/commons/groups/compbio/projects/CZI_endom/RNA_temp/c2s_deconv/CZI_c2s_embeddings_lineage_test_pseudobulk_raw_genename_27B.csv \
--signature /gpfs/commons/groups/compbio/projects/CZI_endom/RNA_temp/c2s_deconv/CZI_c2s_embeddings_lineage_train_signature_matrix_for_nnls_genename_27B.csv \
--output /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/test/nnls_cell_proportions_c2s_27B.csv