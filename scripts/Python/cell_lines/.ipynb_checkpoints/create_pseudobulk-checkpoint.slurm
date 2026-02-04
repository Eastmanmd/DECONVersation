#!/bin/bash
#SBATCH --job-name=create_pseudobulk_cell_lines
#SBATCH --output=logs/create_pseudobulk_cell_lines.log   
#SBATCH --error=logs/create_pseudobulk_cell_lines.err   
#SBATCH --time=72:00:00
#SBATCH --mem=64G                     
#SBATCH --partition=cpu  


# Run the script
# sample_ids GSM6807287_M1 GSM6807288_M2 GSM6807289_M3 GSM6807290_M4

/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/lineage/create_pseudobulk.py \
--adata_path /gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/GSE220608/geneformer/GSM6807287_cleanso_for_geneformer.h5ad \
--group_var orig.ident \
--celltype_var type \
--sample_ids GSM6807287_M1 GSM6807288_M2 GSM6807289_M3 GSM6807290_M4 \
--combine_samples True \
--n_cells_per_pseudobulk 1500 \
--n_pseudobulks 4000 \
--log_file /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/cell_lines/scripts/log_files/create_pseudobulk_merged.log \
--out_pseudobulk_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/train/pseudobulk_v3.csv \
--out_prop_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/train/cell_proportions_v3.csv