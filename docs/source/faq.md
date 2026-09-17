# FAQ

## Which foundation models are supported?

Geneformer, Cell2Sentence, CellHermes, and scGPT, plus PCA and scVI for comparison.

## Why can't I install all the scFMs in one environment?

Their dependency pins conflict irreconcilably — they will never be compatible in a
single Python environment. DECONVersation works around this by detecting and only
loading the model(s) available in the current environment. Install one environment
per model using the YAML files in `envs/`:

```bash
conda env create -f deconv_gf.yml
```

Apptainer `.def` files are also provided. See [Installation](installation.md).

## What is the fastest way to get started?

From Python:

```python
import deconversation

res = deconversation.deconverse(
    bulk_df="bulk_rna.csv",
    sig_df="signature_matrix.csv",
    model="path_to/Geneformer-V2-316M",
    temp_output_dir="temp",
)
```

Or from the command line:

```bash
deconverse --bulk bulk_rna.csv --sig signature_matrix.csv --model path_to_model -o results.csv
```

To try it without your own data:

```bash
deconverse --demo --model path_to_model
```

## What input files do I need?

A bulk expression table (CSV) and either a signature matrix (CSV) or a single-cell
reference (`.h5ad`) from which one can be built. When passing a reference directly,
name the cell type column:

```bash
deconverse --bulk bulk_rna.csv --adata reference.h5ad --cell_type_col celltype --model path_to_model
```

## Do my genes need to be symbols or Ensembl IDs?

Geneformer requires Ensembl IDs, so `adata.var.index` must hold Ensembl IDs rather
than gene symbols. Cell2Sentence, scGPT and cellHermes all accept gene symbols. Convert in either direction with:

```python
from deconversation import preprocessing

preprocessing.gene_id_name_map(...)
```

## How do I build a signature matrix?

`preprocessing.create_signature_matrix()` averages gene expression across cell types,
producing a gene-by-cell-type table covering every unique cell type in the reference
dataset.

## Which solvers are available?

`nnls`, `dwls` (solver only), `ridge`, `elasticnet`, and `nusvr`.

`deconvolution.run_deconv()` uses NNLS by default;
`deconvolution.run_all_deconv()` runs every available solver so you can compare them.

## Embedding extraction left files behind. What are they?

Extraction writes intermediates such as the tokenized dataset. Set
`delete_temp_files=True` to remove them automatically once extraction finishes, or
`False` to keep them for inspection or reuse.

## Do I need a GPU?

While embeddings can be extracted from some foundation models with just the CPU, it is much slower.

## How do I test the workflow without real bulk data?

Use `pseudobulk.generate_pseudobulk()` to synthesize pseudobulk samples from a
single-cell reference, along with the ground-truth proportion matrix. You control
the number of samples, the proportion bounds, cells per sample, and `random_state`
for reproducibility. See the
[pseudobulk tutorial](tutorials/run_deconversation_on_pseudobulk_data.ipynb).

## How is performance evaluated?

Against ground truth using RMSE and Pearson correlation.
`visualization.visualize_solvers()` plots both across all solvers used; set
`level="cell_type"` to break the metrics down per cell type.
`visualization.plot_true_vs_predicted()` plots predicted against true proportions,
with `stratify_by_celltype=True` for one panel per cell type.


## Can I compare against a non-foundation-model baseline?

Yes. `embeddings.extract_components()` computes PCA embeddings, fit on the
signature matrix and applied to the bulk data, so you can run the identical
pipeline on PCA instead. Note that this caps the number of components at the number
of cell types in the signature matrix. scVI is also supported for comparison.

