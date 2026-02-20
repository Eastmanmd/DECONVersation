#!/bin/bash
#SBATCH --job-name=run_nnls_emb_mode_finetuned
#SBATCH --output=logs/run_nnls_emb_mode_finetuned.log   
#SBATCH --error=logs/run_nnls_emb_mode_finetuned.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=cpu  

# Run script
/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/Python/run_nnls_across_emb_mode_layers.py 