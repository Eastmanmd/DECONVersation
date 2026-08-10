from __future__ import annotations
from typing import BinaryIO
from io import BytesIO
import pandas as pd

try:
    # Python 3.9+
    from importlib.resources import files
except ImportError:
    # Python 3.8
    from importlib_resources import files

_PACKAGE = "deconversation"
_RESOURCE_ROOT = files(_PACKAGE).joinpath("resources")
EXAMPLES = files("deconversation").joinpath("examples")

def open_gene_name_mapping() -> BinaryIO:
    """
    Open the bundled gene-name-to-Ensembl mapping as a binary stream.

    The caller is responsible for closing the returned stream, preferably
    by using it as a context manager.
    """
    resource = _RESOURCE_ROOT.joinpath(
        "gene_symbol_ensembl_id_map.csv"
    )

    if not resource.is_file():
        raise FileNotFoundError(
            "The packaged gene-name mapping resource is missing: "
            "resources/gene_symbol_ensembl_id_map.csv"
        )

    return resource.open("rb")

def load_example_csv(filename: str) -> pd.DataFrame:
    allowed = {
        "demo_bulk.csv",
        "demo_sig_mat.csv",
    }

    if filename not in allowed:
        raise ValueError(
            f"Unknown example {filename!r}. "
            f"Available examples: {sorted(allowed)}"
        )

    resource = EXAMPLES.joinpath(filename)

    if not resource.is_file():
        raise FileNotFoundError(
            f"Packaged example is missing: {filename}"
        )

    # Reading bytes works even when the package is not represented by an
    # ordinary directory on disk.
    return pd.read_csv(BytesIO(resource.read_bytes()),index_col = 0)
