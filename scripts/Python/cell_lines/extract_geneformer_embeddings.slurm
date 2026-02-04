#!/bin/bash
#SBATCH --job-name=extract_embed
#SBATCH --output=logs/extract_embed.log   
#SBATCH --error=logs/extract_embed.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=gpu
#SBATCH --gres=gpu  

# Run the script
/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/cell_lines/scripts/extract_geneformer_embeddings.py
