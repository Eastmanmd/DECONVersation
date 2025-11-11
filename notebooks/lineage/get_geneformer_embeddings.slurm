#!/bin/bash
#SBATCH --job-name=get_gf_embed
#SBATCH --output=logs/get_gf_embed.log   
#SBATCH --error=logs/get_gf_embed.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=gpu  
#SBATCH --gres=gpu 


# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/notebooks/celltype/get_geneformer_embeddings.py \
--input_csv /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/test/pseudobulk_raw.csv \
--token_output_dir /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/temp/tokenize \
--token_output_name temp_tokens \
--embedding_output_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_celltype/temp/temp_embeddings.csv