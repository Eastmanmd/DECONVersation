#!/bin/bash
#SBATCH --job-name=extract_embed_hoek
#SBATCH --output=logs/extract_embed_hoek.log   
#SBATCH --error=logs/extract_embed_hoek.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=gpu
#SBATCH --gres=gpu 
 

# Run the script
/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/Python/extract_geneformer_embeddings_hoek.py

