# DECONVersation
<img src="images/deconversation_png.png" width="400">

DECONVersation is a tool designed for the deconvolution of bulk RNA-seq data using embeddings derived from large-scale, LLM-based foundation models. DECONVersation produces robust  cell type proportions by leveraging these high-dimensional embeddings to mitigate batch effects typically present in single-cell reference signature matrices.

---

## Overview

This project provides:

- Installation guide for DECONVersation
- Step-by-step tutorials for embedding extraction and downstream deconvolution analysis
- Sample pseudobulk dataset for deconvolution testing 
- Perfomance evaluation comparing estimates from DECONVersation and other deconvolution tools
- Guide to finetuning foundational models using DECONVersation (Geneformer and Cell2Sentence)

---

## DECONVersation Features

- Functions to extract geneformer and cell2sentence embeddings from a given bulk/pseudobulk dataset
- Functions to create a signature matrix and extract embeddings given a single cell reference data (.h5ad)
- Functions to estimate cell types using NNLS from the bulk and signature matrix embeddings 
- Evaluation of predicted cell type proportions against ground truth
- Performance metrics (RMSE, Pearson correlation) 
- Visualization of method performance (RMSE vs Correlation plots)
- Functions to create pseudobulk data for deconvolution is testing. 

---

## Installation

### 1. …
