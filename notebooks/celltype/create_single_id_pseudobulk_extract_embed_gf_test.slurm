#!/bin/bash
#SBATCH --job-name=gf_deconv_one_test
#SBATCH --output=logs/gf_deconv_one_test.log   
#SBATCH --error=logs/gf_deconv_one_test.err   
#SBATCH --time=72:00:00
#SBATCH --mem=64G                     
#SBATCH --partition=gpu  
#SBATCH --gres=gpu 


# Run the script
# n_cells_per_pseudobulk 40

/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/celltype/create_pseudobulk_extract_embed_gf.py \
--adata_path /gpfs/commons/groups/compbio/projects/CZI_endom/RNA_temp/62_harmony_102224_Seuratv34_newannot_counts.h5ad \
--heca_path /nfs/home/rfu/projects/CZI_endom/CZI_HECAtype.csv.gz \
--gene_map_path /gpfs/commons/groups/compbio/projects/CZI_endom/RNA_temp/gene_names_gene_ids_czi_endo.csv \
--sample_ids B15-CZI251N \
--out_prop_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/single_samp/test/ \
--out_adata_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/single_samp/test/test_set.h5ad \
--token_output_dir /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/single_samp/test/tokenize \
--token_output_name token_test_set \
--random_state 10 \
--embedding_output_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/single_samp/test/gf_zeroshot_pseudobulk_test_set.csv


 