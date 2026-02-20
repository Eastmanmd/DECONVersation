#!/bin/bash
#SBATCH --job-name=get_gf_embed_sig_mat
#SBATCH --output=logs/get_gf_embed_sig_mat.log   
#SBATCH --error=logs/get_gf_embed_sig_mat.err   
#SBATCH --time=72:00:00
#SBATCH --mem=16G                     
#SBATCH --partition=gpu  
#SBATCH --gres=gpu 


# Run script  

/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/scripts/Python/get_geneformer_embeddings.py \
--input_csv /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/pseudo_bulk/pb_sig_matrix.csv \
--token_output_dir /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data_hoek/train/tokenize_sig_mat \
--geneformer_model_path /gpfs/commons/groups/compbio/projects/CZI_endom/geneformer/output/czi_perturb_noStemPro/rand_undersamp_v2_316M_1500/251103_geneformer_cellClassifier_cell_annot_subset/ksplit1/ \
--token_output_name temp_tokens \
--embedding_output_path /gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv_data/pseudo_bulk/gf_czi_sig_mat_pseudobulk_embed_finetuned_cls_emb_0.csv