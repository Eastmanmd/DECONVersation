#!/bin/bash
#SBATCH --job-name=create_pseudobulk
#SBATCH --output=logs/create_pseudobulk.log   
#SBATCH --error=logs/create_pseudobulk.err   
#SBATCH --time=72:00:00
#SBATCH --mem=64G                     
#SBATCH --partition=cpu  


# Run the script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/lineage/create_pseudobulk.py \
--adata_path /path/to/single_cell_data.h5ad \
--group_var id \
--sample_ids B8-CZI026N B4P4-CZI04N \
--celltype_var heca_lineage \
--n_cells_per_pseudobulk 800 \
--out_pseudobulk_path /path/to/output/pseudobulk.csv \
--out_prop_path /path/to/output/cell_proportions.csv