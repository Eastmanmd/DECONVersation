#!/bin/bash
#SBATCH --job-name=get_gf_embed_test_bulk
#SBATCH --output=logs/get_gf_embed_test_bulk.log   
#SBATCH --error=logs/get_gf_embed_test_bulk.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=gpu  
#SBATCH --gres=gpu 


# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/Python/get_geneformer_embeddings.py \
--input_csv /gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/GSE220608/GSE220605_DataMatrix_Mixture_CellLines_Bulk_RNA-seq_gene_ids.csv \
--token_output_dir /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/test/tokenize_bulk \
--token_output_name temp_tokens \
--embedding_output_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/test/gf_embeddings_cell_lines_bulk.csv