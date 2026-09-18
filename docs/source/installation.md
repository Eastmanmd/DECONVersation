# Installation

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
