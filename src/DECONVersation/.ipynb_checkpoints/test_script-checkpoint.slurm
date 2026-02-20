#!/bin/bash
#SBATCH --job-name=test_script_c2s
#SBATCH --output=test_script_c2s.log   
#SBATCH --error=test_script_c2s.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=cpu


/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/src/DECONVersation/test_script.py