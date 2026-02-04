#!/bin/bash
#SBATCH --job-name=get_scVI_embed_test
#SBATCH --output=logs/get_scVI_embed_test.log   
#SBATCH --error=logs/get_scVI_embed_test.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=cpu

# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/lineage/get_scVI_embeddings.py \
--input_csv /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/test/pseudobulk_raw.csv \
--scvi_model_path /gpfs/commons/groups/compbio/projects/ao_projects/scvi-human-2024-07-01 \
--embedding_output_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_lineage/test/scVI_pseudobulk_test_set.csv