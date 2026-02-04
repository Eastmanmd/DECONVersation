#!/bin/bash
#SBATCH --job-name=fine_tune_gf_2000
#SBATCH --output=logs/fine_tune_gf_2000.log   
#SBATCH --error=logs/fine_tune_gf_2000.err   
#SBATCH --time=72:00:00
#SBATCH --mem=32G                     
#SBATCH --partition=gpu  
#SBATCH --gres=gpu 
 


# Load Python module if needed
# module load scgpt/2.0

# Run the script
/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/Python/finetune_geneformer_model.py