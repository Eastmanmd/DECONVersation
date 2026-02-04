#!/bin/bash
#SBATCH --job-name=runMusicCellLines         
#SBATCH --output=logs/runMusicCellLines.out 
#SBATCH --error=logs/runMusicCellLines.err   
#SBATCH --mem=64G

# Run your R script
Rscript /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/R/cell_lines_pseudobulk_vs_cell_lines_ref.R