```{image} _static/deconversation.png
:alt: DECONVersation
:width: 500px
:align: center
```

# Welcome to DECONVersation's documentation!

[![Documentation Status](https://readthedocs.org/projects/deconversation/badge/?version=latest)](https://deconversation.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/DECONVersation.svg)](https://pypi.org/project/DECONVersation/)
[![Python versions](https://img.shields.io/pypi/pyversions/DECONVersation.svg)](https://pypi.org/project/DECONVersation/)

Deconvolution of bulk RNA-seq data using embeddings from single-cell foundation models.

## Introduction

DECONVersation leverages embedding representations from large-scale, LLM-based
foundation models to perform deconvolution of bulk RNA-seq data. This takes
advantage of the strengths of scFMs in faithfully representing transcriptomes,
learning meaningful biological networks, and minimizing batch effect and noise.
Currently, cell embeddings from Geneformer, Cell2Sentence, CellHermes, and scGPT
are supported (+PCA and scVI for comparison).

DECONVersation enables end-to-end deconvolution through a set of easy-to-use
functions. Embeddings can be extracted from both bulk and single-cell datasets,
with single-cell embeddings used to construct robust signature matrices from
`.h5ad` references. Cell type proportions are then estimated via NNLS directly in
embedding space. Built-in benchmarking tools evaluate predictions against ground
truth using RMSE and Pearson correlation, complemented by visualization utilities
for assessing method performance. DECONVersation also supports testing and
validation with in-built pseudobulk functions.

## Installation

While DECONVersation itself is lightweight and easy to install with
`pip install DECONVersation`, the various single-cell foundation models themselves
are not. In fact, due to dependency restrictions, they will never be compatible in
the same Python environment. DECONVersation works around this by detecting and
only loading the available model(s).

For each scFM model and package, users should consult the corresponding official
installation guides. We also provide conda environment YAML files in the `envs`
directory that are reproducibly operational on our hardware (NVIDIA L40S), each
corresponding to one of the scFMs plus DECONVersation:

```bash
conda env create -f deconv_gf.yml
```

Apptainer `.def` files are also included.

```{toctree}
:maxdepth: 2
:caption: Getting Started

introduction
installation
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/run_deconversation_on_bulk_geneformer
tutorials/run_deconversation_on_pseudobulk_data
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api
```

```{toctree}
:maxdepth: 1
:caption: References

faq
references
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`



