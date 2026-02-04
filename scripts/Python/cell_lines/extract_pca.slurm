#!/bin/bash
#SBATCH --job-name=extract_pca
#SBATCH --output=logs/extract_pca.log   
#SBATCH --error=logs/extract_pca.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=cpu  


# Run the script

/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python  /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/cell_lines/scripts/extract_pca.py \
    --train_pseudobulk /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/train/pseudobulk.csv \
    --test_pseudobulk /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/test/pseudobulk.csv \
    --signature_matrix /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/train/signature_matrix_for_nnls.csv \
    --outdir /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/train/ \
    --outdir_test /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/test/ \
    --n_components 100