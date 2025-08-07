#!/bin/bash
#SBATCH --job-name=runCziPseudo          
#SBATCH --output=logs/runCziPseudo.out 
#SBATCH --error=logs/runCziPseudo.err   
#SBATCH --mem=128G

# Run your R script
Rscript ~/scripts/POLY/P1000/scripts/music_czi_pseudobulk_vs_czi_ref.R