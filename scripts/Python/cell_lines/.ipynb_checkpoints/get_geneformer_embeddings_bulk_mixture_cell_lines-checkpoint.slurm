#!/bin/bash
#SBATCH --job-name=get_gf_embed_mix
#SBATCH --output=logs/get_gf_embed_mix.log   
#SBATCH --error=logs/get_gf_embed_mix.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=gpu  
#SBATCH --gres=gpu 


# Run script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/Python/get_geneformer_embeddings.py \
--input_csv /gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/GSE220608/GSE220605_DataMatrix_Mixture_CellLines_Bulk_RNAseq_gene_ids_mixtures_only.csv \
--token_output_dir /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/temp/tokenize \
--token_output_name temp_tokens_mixture \
--embedding_output_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_cell_lines/temp/gf_embed_mixture_cell_lines.csv