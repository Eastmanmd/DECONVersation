#!/bin/bash
#SBATCH --job-name=runCziPseudoCelltype         
#SBATCH --output=logs/runCziPseudoCelltype.out 
#SBATCH --error=logs/runCziPseudoCelltype.err   
#SBATCH --mem=64G

# Run your R script
Rscript /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/music_czi_pseudobulk_vs_czi_ref_celltype.R