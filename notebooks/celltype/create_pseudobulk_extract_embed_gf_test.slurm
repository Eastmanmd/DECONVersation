#!/bin/bash
#SBATCH --job-name=gf_deconv_celltype_test
#SBATCH --output=logs/gf_deconv_celltype_test.log   
#SBATCH --error=logs/gf_deconv_celltype_test.err   
#SBATCH --time=72:00:00
#SBATCH --mem=64G                     
#SBATCH --partition=gpu  
#SBATCH --gres=gpu 


# Run the script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/celltype/create_pseudobulk_extract_embed_gf.py \
--adata_path /gpfs/commons/groups/compbio/projects/CZI_endom/RNA_temp/62_harmony_102224_Seuratv34_newannot_counts.h5ad \
--heca_path /nfs/home/rfu/projects/CZI_endom/CZI_HECAtype.csv.gz \
--gene_map_path /gpfs/commons/groups/compbio/projects/CZI_endom/RNA_temp/gene_names_gene_ids_czi_endo.csv \
--sample_ids B8-CZI026N B4P4-CZI04N \
--out_prop_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/test/ \
--out_adata_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/test/test_set.h5ad \
--token_output_dir /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/test/tokenize \
--token_output_name token_test_set \
--embedding_output_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/test/gf_zeroshot_pseudobulk_test_set.csv