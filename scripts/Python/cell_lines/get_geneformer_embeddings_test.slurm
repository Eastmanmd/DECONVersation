#!/bin/bash
#SBATCH --job-name=get_gf_embed_test
#SBATCH --output=logs/get_gf_embed_test.log   
#SBATCH --error=logs/get_gf_embed_test.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=gpu  
#SBATCH --gres=gpu 


# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/lineage/get_geneformer_embeddings.py \
--input_csv /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/test/pseudobulk_counts.csv \
--token_output_dir /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/test/tokenize \
--token_output_name temp_tokens \
--embedding_output_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/cell_lines/test/gf_pseudobulk_embed.csv