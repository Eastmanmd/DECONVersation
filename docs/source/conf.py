import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../.."))

project = "DECONVersation"
copyright = f"{date.today().year}, DECONVersation contributors"
author = "DECONVersation contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_nb",
]

# Heavy deps that must not be installed on the RTD builder
autodoc_mock_imports = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "scanpy",
    "geneformer",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Ship notebooks with outputs already saved; the builder has no GPU or data
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
}

html_theme = "sphinx_rtd_theme"
html_title = "DECONVersation"
html_static_path = ["_static"]
html_theme_options = {
    "logo_only": True,
    "navigation_depth": 3,
    "collapse_navigation": False,
}

