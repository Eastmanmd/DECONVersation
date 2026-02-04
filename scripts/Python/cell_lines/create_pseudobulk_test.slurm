#!/bin/bash
#SBATCH --job-name=create_pseudobulk_cell_lines_test
#SBATCH --output=logs/create_pseudobulk_cell_lines_test.log   
#SBATCH --error=logs/create_pseudobulk_cell_lines_test.err   
#SBATCH --time=72:00:00
#SBATCH --mem=64G                     
#SBATCH --partition=cpu  


# Run the script
# sample_ids GSM6807291_M5 GSM6807292_M6 \

/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/lineage/create_pseudobulk.py \
--adata_path /gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/GSE220608/geneformer/GSM6807287_cleanso_for_geneformer.h5ad \
--group_var orig.ident \
--celltype_var type \
--n_cells_per_pseudobulk 500 \
--sample_ids GSM6807291_M5 GSM6807292_M6 \
--out_pseudobulk_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/test/pseudobulk.csv \
--out_prop_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/test/cell_proportions.csv