# Quick Start

```python
# python
import deconversation
res = deconversation.deconverse(
    bulk_df = "bulk_rna.csv",
    sig_df = "signature_matrix.csv", # or adata = "reference.h5ad", needs one or the other
    model = "path_to/Geneformer-V2-316M",
    temp_output_dir = "temp"
)
```

```bash
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
