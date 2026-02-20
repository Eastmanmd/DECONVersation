#!/bin/bash
#SBATCH --job-name=test_emb_modes_sig_mat_finetuned
#SBATCH --output=logs/test_emb_modes_sig_mat_finetuned.log   
#SBATCH --error=logs/test_emb_modes_sig_mat_finetuned.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=gpu  
#SBATCH --gres=gpu 


# Run script
/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/Python/layer_quant_emb_mode_effects.py