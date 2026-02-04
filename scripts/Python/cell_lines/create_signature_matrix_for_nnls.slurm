#!/bin/bash
#SBATCH --job-name=create_sig_mat
#SBATCH --output=logs/create_sig_mat.log   
#SBATCH --error=logs/create_sig_mat.err   
#SBATCH --time=72:00:00
#SBATCH --mem=64G                     
#SBATCH --partition=cpu  

# Run the script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/lineage/create_signature_matrix_for_nnls.py \
--adata_path /gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/GSE220608/geneformer/GSM6807287_cleanso_for_geneformer.h5ad \
--sample_col orig.ident \
--sample_ids GSM6807287_M1 GSM6807288_M2 GSM6807289_M3 GSM6807290_M4 \
--celltype_col type \
--celltypes T47D BT474 MCF7 Jurkat Thp1 hMSC \
--groupby type \
--output_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/train/signature_matrix_for_nnls.csv