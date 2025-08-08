#!/bin/bash
#SBATCH --job-name=runCziPseudo          
#SBATCH --output=logs/runCziPseudo.out 
#SBATCH --error=logs/runCziPseudo.err   
#SBATCH --mem=128G

# Run your R script
Rscript /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/music_czi_pseudobulk_vs_czi_ref.R