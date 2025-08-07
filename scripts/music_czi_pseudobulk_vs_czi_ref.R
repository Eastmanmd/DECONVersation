#Load libraries.
#Do module purge;module load R/4.4.1, module load seurat/4.0.1 before this.
library(Seurat)
library(MuSiC)
library(SummarizedExperiment)
library(tidyr)
library(dplyr)
library(ggplot2)
library(here)
library(stringr)


# Load bulk counts.
genes_counts <- read.csv("/gpfs/commons/groups/compbio/projects/ao_projects/ml_deconv/data/pseudobulk_test_set.csv", header = T, check.names = F)


#Convert gene counts to integer (from RSEM so originally was not).
for(i in 1:ncol(genes_counts))
{
  genes_counts[,i] <- as.integer(genes_counts[,i])
}

# convert ensembl IDs to gene names
gene_ids <- readr::read_csv("/gpfs/commons/groups/compbio/projects/CZI_endom/RNA_temp/gene_names_gene_ids_czi_endo.csv")

gene_id_dict <- gene_ids %>%
  select(gene_id, gene_name) %>%
  distinct() %>%
  drop_na()

id_to_symbol <- setNames(gene_id_dict$gene_name, gene_id_dict$gene_id)

colnames(genes_counts) <- unname(id_to_symbol[colnames(genes_counts)])


# Convert to matrix
all_bulk_samples <- genes_counts %>% t()
rm(genes_counts)

all_bulk_samples <- as.matrix(all_bulk_samples)

#Read in single cell reference Seurat object, convert to SingleCellExperiment object.
#singlespinal.sce <- readRDS("/gpfs/commons/groups/compbio/projects/CZI_endom/publicData/HECA/endometriumAtlasV2_cells_nuclei_merged.seurat.RDS")
singlespinal.sce  <- readRDS("/gpfs/commons/groups/compbio/projects/CZI_endom/RNA_temp/62_harmony_102224_Seuratv34_newannot.rds")

# Get the HECA annotations for the czi object'
heca_annot <- readr::read_csv("/nfs/home/rfu/projects/CZI_endom/CZI_HECAtype.csv.gz")

# Check if cell IDs match
# Assume your Seurat object is named `seurat_obj`
matching <- colnames(singlespinal.sce) == heca_annot$cell
print(table(matching))

# Add annotations to Seurat object metadata
singlespinal.sce$heca_celltype <- heca_annot$celltype
singlespinal.sce$heca_lineage <- heca_annot$lineage


# Get common genes
common_genes <- intersect(rownames(all_bulk_samples), rownames(singlespinal.sce))

all_bulk_samples <- all_bulk_samples[common_genes, ]
singlespinal.sce <- singlespinal.sce[common_genes, ]

# Convert to SingleCellExperiment object
singlespinal.sce <-  as.SingleCellExperiment(singlespinal.sce)

#Run Music.
#Here "projid" is a proxy for patient/sample in the single cell data.
results <- music_prop(bulk.mtx = all_bulk_samples,
                      sc.sce = singlespinal.sce,
                      clusters = 'newtype',
                      samples = 'id',
                      verbose=TRUE)
#clusters: newlineage
#sample: id


#Save.
#Saves a list where first item named "Est.prop.weighted" is the proportion matrix.
#saveRDS(results,file="~/scripts/POLY/P1000/music/music_prop_endo_deconvolution_vs_endo_reference_with_uterus_lineage_no_nuclei_v3_1.rds")

#Write proportion matrix to CSV.
write.csv(results[["Est.prop.weighted"]],file="~/scripts/POLY/P1000/data/pseudobulk/music_prop_pseudobulk_czi_vs_czi_single_cell_ref_estimated_newtype.csv")
#write.csv(results[["Est.prop.weighted"]],file="~/scripts/POLY/P1000/music/music_prop_endo_deconvolution_vs_endo_reference_with_uterus_lineage_no_nuclei_v3_1.csv")

