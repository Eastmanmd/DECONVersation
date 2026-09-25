<h1 align="left">
  <img src="https://raw.githubusercontent.com/Eastmanmd/DECONVersation/main/docs/deconversation.png" width="500">
</h1>

[![PyPI version](https://img.shields.io/pypi/v/DECONVersation.svg)](https://pypi.org/project/DECONVersation/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Downloads](https://static.pepy.tech/personalized-badge/deconversation?period=total&units=none&left_text=downloads&left_color=grey&right_color=blue)](https://pypi.org/project/DECONVersation/)
[![Github action](https://github.com/Eastmanmd/DECONVersation/actions/workflows/python-app.yml/badge.svg)](https://github.com/Eastmanmd/DECONVersation/tree/main/envs)
[![Zenodo](https://img.shields.io/badge/Zenodo-DOI-f5de53?&color=f5de53)](https://zenodo.org/records/22949405)

DECONVersation leverages embedding representations from large-scale, LLM-based foundation models to perform deconvolution of bulk RNA-seq data. This takes advantage of the strengths of scFMs in faithfully representing transcriptomes, learning meaningful biological networks, and minimizing batch effect and noise. Currently, cell embeddings from [Geneformer](https://huggingface.co/ctheodoris/Geneformer), [Cell2Sentence](https://github.com/vandijklab/cell2sentence), [CellHermes](https://github.com/theislab/CellHermes), and [scGPT](https://github.com/bowang-lab/scGPT) are supported (+PCA and scVI for comparison). 

DECONVersation enables end-to-end deconvolution through a set of easy-to-use functions. Embeddings can be extracted from both bulk and single-cell datasets, with single-cell embeddings used to construct robust signature matrices from .h5ad references. Cell type proportions are then estimated via NNLS directly in embedding space. Built-in benchmarking tools evaluate predictions against ground truth using RMSE and Pearson correlation, complemented by visualization utilities for assessing method performance. DECONVersation also supports testing and validation with in-built [pseudobulk functions](tutorials/run_deconversation_on_pseudobulk_data.ipynb), [model fine-tuning](tutorials/finetune_geneformer_for_cell_type_classification.ipynb) and [attention extraction](tutorials/extract_attention_weights.ipynb).

<h1 align="left">
  <img src="https://raw.githubusercontent.com/Eastmanmd/DECONVersation/main/docs/workflow.png" width="900">
</h1>

---

## Benchmarking 

DECONVersation was benchmarked across 6 real bulk RNA-seq datasets with ground truths and 2 pseudobulk dataset, spanning diverse tissue types and experimental conditions, to evaluate deconvolution performance and generalizability.

<h1 align="left">
  <img src="https://raw.githubusercontent.com/Eastmanmd/DECONVersation/main/docs/full_bench.png" width="900">
</h1>

<b> Summary </b> <br>
Across 6 benchmarked real bulk and 2 pseudobulk datasets, we calculate overall RMSE and correlation coefficient alongside mean RMSE and correlation averaged across cell types. Fine-tuned Cell2Sentence and Geneformer-based embeddings both demonstrate consistent deconvolution performance across all datasets, with fine-tuned models outperforming their zero-shot counterparts in each case. Though zero-shot performance is already comparable to some common tools in the field, this highlights the benefit of fine-tuning (training models to predict cell type annotations from a single-cell reference). Among the tested tools, only DWLS R package achieves comparable performance to the fine-tuned embedding-based approaches available in DECONVersation.

| # | Dataset | Source | Ground Truth | Cell Type # | 
| -------- | -------- | --------  | --------  | --------  |
| 1 | [PBMC (Hoek)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118528)| PBMC | FACS | 5 |
| 2 | [PBMC (Finotello)](https://pubmed.ncbi.nlm.nih.gov/31126321/)| PBMC | FACS | 5 |
| 3 | [PBMC (Morandini)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10828344/)| PBMC | FACS | 5 |
| 4 | [Cell Line Mixture (Cobos)](https://europepmc.org/article/med/37528411)| Cell Line Mixture| Mixture Count | 6 |
| 5 | [Pre-Frontal Cortex (Huuki-Myers)](https://pubmed.ncbi.nlm.nih.gov/38781370/)| DLPFC | RNAScope/IF | 6 |
| 6 | [Retina (Guo)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11789644/)| Retina | snRNA | 6 |
| 7 | [HECA (Marečková)](https://pubmed.ncbi.nlm.nih.gov/39198675/) | Endometrium | pseudobulk | 9 |
| 8 | [Lung tumor (Guimarães)](https://pubmed.ncbi.nlm.nih.gov/38972873/) | Lung | pseudobulk | 10 |

---

## Installation
While DECONVersation itself is lightweight and easy to install with `pip install DECONVersation`, the various single cell foundation models themselves are not. In fact, due to dependency restrictions, they will never be compatible in the same python environment. DECONVersation works around this by detecting and only loading the available model(s). For each scFM model and package, users should consult the corresponding official installation guides. 

We also provide conda env yaml files in the `envs` directory that are reproducibly operational on our hardware (NVIDIA L40S), each corresponding to one of the scFMs + DECONVersation. They can be installed with e.g. `conda env create -f deconv_gf.yml`. Apptainer def files are also included. Alternatively, we provide prebuilt CUDA12.9 singularity images for each scFM, with DECONVersation v0.1.0 installed, on [Zenodo](https://zenodo.org/records/22949405).

---

## Quick start
```
# python
import deconversation
res = deconversation.deconverse(
    bulk_df = "bulk_rna.csv",
    sig_df = "signature_matrix.csv", # or adata = "reference.h5ad", needs one or the other
    model = "path_to/Geneformer-V2-316M",
    temp_output_dir = "temp"
)

# command line
deconverse --help
deconverse --demo --model path_to_model
deconverse --bulk bulk_rna.csv --sig signature_matrix.csv --model path_to_model -o deconv_results.csv
deconverse --bulk bulk_rna.csv --adata reference.h5ad --cell_type_col celltype --model path_to_model

# with SIF mounting
singularity exec --nv \
  --bind path_to_input_dir:/data:ro \
  --bind path_to_output_dir:/results \
  --bind path_to_model:/model:ro \
  deconversation-geneformer.sif \
  deconverse \
    --bulk /data/bulk_rna.csv \
    --sig /data/signature_matrix.csv \
    --mode geneformer \
    --model /model \
    --temp-output-dir /results/temp \
    --output /results/proportions.csv
```
---

## Tutorials

- [DECONVersation on bulk RNA-seq using Geneformer](tutorials/run_deconversation_on_bulk_geneformer.ipynb): Extract embeddings and deconvolute on bulk against a single cell reference.
- [DECONVersation on pseudobulk using Geneformer](tutorials/run_deconversation_on_pseudobulk_data.ipynb): Validate deconvolution using pseudobulk data.
- [Fine-tuning geneformer for cell type classification](tutorials/finetune_geneformer_for_cell_type_classification.ipynb): Fine-tune geneformer for cell type classification 
- [Extracting attention weights using DECONVersation](tutorials/extract_attention_weights.ipynb): Extract scFM attention weights (Geneformer, Cell2Sentence & cellHermes)

---

## Suggested Reading
- [Geneformer](https://www.nature.com/articles/s41586-023-06139-9) Transfer learning enables predictions in network biology
- [Cell2Sentence](https://pmc.ncbi.nlm.nih.gov/articles/PMC11565894/) Cell2Sentence: Teaching Large Language Models the Language of Biology
- [CellHermes](https://www.biorxiv.org/content/10.1101/2025.11.07.687322v1) Language may be all omics needs: Harmonizing multimodal data for omics understanding with CellHermes
- [scGPT](https://www.nature.com/articles/s41592-024-02201-0) scGPT: toward building a foundation model for single-cell multi-omics using generative AI
---
<a href="https://www.flaticon.com/free-icons/robot" title="robot icons">Robot icons created by Hilmy Abiyyu A. - Flaticon</a>
