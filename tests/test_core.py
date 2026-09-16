from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="core currently imports the torch-backed solver eagerly")

from deconversation.core import deconverse
from deconversation.resource_loader import load_example_csv


def test_packaged_examples_load():
    bulk = load_example_csv("demo_bulk.csv")
    signature = load_example_csv("demo_sig_mat.csv")

    assert not bulk.empty
    assert not signature.empty
    assert bulk.index.intersection(signature.index).size > 0


def test_unknown_packaged_example_is_rejected():
    with pytest.raises(ValueError, match="Unknown example"):
        load_example_csv("not-an-example.csv")


def test_raw_demo_runs_with_existing_output_directory(tmp_path):
    result = deconverse(
        bulk_df="unused",
        sig_df="unused",
        model=None,
        mode="raw",
        demo=True,
        temp_output_dir=str(tmp_path),
    )

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert np.isfinite(result.to_numpy()).all()
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-8)
    assert (tmp_path / "bulk_embedding.csv").is_file()


def test_pca_demo_runs_without_foundation_model(tmp_path):
    result = deconverse(
        bulk_df="unused",
        sig_df="unused",
        model=None,
        mode="pca",
        demo=True,
        temp_output_dir=str(tmp_path),
    )

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert np.isfinite(result.to_numpy()).all()


def test_raw_demo_creates_output_directory(tmp_path):
    output_dir = tmp_path / "new" / "nested"

    deconverse(
        bulk_df="unused",
        sig_df="unused",
        model=None,
        mode="raw",
        demo=True,
        temp_output_dir=str(output_dir),
    )

    assert output_dir.is_dir()
    assert (output_dir / "bulk_embedding.csv").is_file()
