# Introduction

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
