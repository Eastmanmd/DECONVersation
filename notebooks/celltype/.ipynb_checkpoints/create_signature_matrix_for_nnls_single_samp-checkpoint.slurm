#!/bin/bash
#SBATCH --job-name=create_sig_mat_single_samp
#SBATCH --output=logs/create_sig_mat_single_samp.log   
#SBATCH --error=logs/create_sig_mat_single_samp.err   
#SBATCH --time=72:00:00
#SBATCH --mem=64G                     
#SBATCH --partition=cpu  

# Run the script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/celltype/create_signature_matrix_for_nnls.py \
--adata_path /gpfs/commons/groups/compbio/projects/CZI_endom/RNA_temp/62_harmony_102224_Seuratv34_newannot_counts.h5ad \
--sample_col id \
--sample_ids B15-CZI251N \
--celltype_col newtype \
--celltypes EpithelialGL_harmonal Endothelial Stromal_EMT EpithelialPino \
--groupby newtype \
--output_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/single_samp/train/signature_matrix_for_nnls.csv