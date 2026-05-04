<h1 align="left">
  <img src="docs/deconversation.png" width="500">
</h1>

DECONVersation is a tool designed for the deconvolution of bulk RNA-seq data using embeddings derived from large-scale, LLM-based foundation models. DECONVersation produces robust  cell type proportions by leveraging these high-dimensional embeddings to mitigate batch effects typically present in single-cell reference signature matrices.

```mermaid
---
config:
  theme: 'neutral'
---
flowchart
    subgraph ide1 [standard]
    direction TB 
    A[scRNA: cell x gene] --> B(full profile: type x gene)
    A[scRNA: cell x gene] --> C(markers)
    D[bulkRNA: sample x gene]
    B --> F(signature: type x marker)
    C --> F
    F bb@==> E
    D db@==> E{{deconv res:
    sample x type%}}
    end
    subgraph ide2 [foundation model]
    direction TB
    A1[scRNA: cell x gene] --> B1(full profile: type x gene)
    A1[scRNA: cell x gene] --> C1([fa:fa-robot finetuned model])
    B1 --> F1(type x embeddings)
    C1 --> F1
    D1[bulkRNA: sample x gene] --> G1(sample x embeddings)
    C1 --> G1
    F1 f1b@==> E1{{deconv res:
    sample x type%}}
    G1 g1b@==> E1
    end

bb@{ curve: linear }
db@{ curve: linear }
f1b@{ curve: linear }
g1b@{ curve: linear }
style A fill:green,color:#fff
style A1 fill:green,color:#fff
style D fill:blue,color:#fff
style D1 fill:blue,color:#fff
style C1 fill:red,color:#fff
style E stroke-width:4px
style E1 stroke-width:4px
```

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

DECONVersation supports end-to-end deconvolution through a set of easy-to-use functions. Geneformer and Cell2Sentence embeddings can be extracted from both bulk and single-cell datasets, with single-cell embeddings used to construct robust signature matrices from .h5ad references. Cell type proportions are then estimated via NNLS directly in embedding space. Built-in benchmarking tools evaluate predictions against ground truth using RMSE and Pearson correlation, complemented by visualization utilities for assessing method performance. DECONVersation also supports testing and validation with in-built pseudobulk functions. 

---

## Tutorials

- [DECONVersation of bulk RNA-seq using Geneformer](tutorials/extracting_embeddings_from_bulk.ipynb): How to extract embeddings (geneformer)and run DECONVersation on bulk using a single cell reference.

---

## Installation

### 1. …


---
