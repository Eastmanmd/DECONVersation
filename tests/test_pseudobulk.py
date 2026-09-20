from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from deconversation.pseudobulk import generate_pseudobulk


def test_pseudobulk_is_reproducible(dense_adata):
    first = generate_pseudobulk(
        dense_adata,
        "cell_type",
        n_pseudobulks=5,
        n_cells_per_pseudobulk=2,
        random_state=17,
    )
    second = generate_pseudobulk(
        dense_adata,
        "cell_type",
        n_pseudobulks=5,
        n_cells_per_pseudobulk=2,
        random_state=17,
    )

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_pseudobulk_proportions_are_valid(dense_adata):
    pseudobulk, proportions = generate_pseudobulk(
        dense_adata,
        "cell_type",
        n_pseudobulks=4,
        n_cells_per_pseudobulk=2,
        random_state=4,
    )

    assert pseudobulk.shape == (4, dense_adata.n_vars)
    assert proportions.shape == (4, 2)
    assert (pseudobulk.to_numpy() >= 0).all()
    assert (proportions.to_numpy() >= 0).all()
    np.testing.assert_allclose(proportions.sum(axis=1), 1.0)


def test_pseudobulk_rejects_missing_cell_type_column(dense_adata):
    with pytest.raises(ValueError, match="not found"):
        generate_pseudobulk(dense_adata, "missing")


def test_empty_anndata_returns_empty_frames():
    empty = ad.AnnData(
        X=np.empty((0, 2)),
        obs=pd.DataFrame({"cell_type": pd.Series(dtype=str)}),
        var=pd.DataFrame(index=["gene_1", "gene_2"]),
    )

    with pytest.warns(UserWarning, match="No cell types"):
        pseudobulk, proportions = generate_pseudobulk(empty, "cell_type")

    assert pseudobulk.empty
    assert proportions.empty


def test_single_rare_population_samples_with_replacement():
    adata = ad.AnnData(
        X=np.array([[1.0, 0.0], [3.0, 2.0]]),
        obs=pd.DataFrame(
            {"cell_type": ["rare", "rare"]},
            index=["cell_1", "cell_2"],
        ),
        var=pd.DataFrame(index=["gene_1", "gene_2"]),
    )

    with pytest.warns(UserWarning, match="replacement"):
        pseudobulk, proportions = generate_pseudobulk(
            adata,
            "cell_type",
            n_pseudobulks=2,
            target_proportion_min=1.0,
            target_proportion_max=1.0,
            n_cells_per_pseudobulk=5,
            random_state=9,
        )

    assert pseudobulk.shape == (2, 2)
    np.testing.assert_allclose(proportions["rare"], 1.0)


def test_default_threshold_does_not_replace_when_enough_cells(dense_adata, recwarn):
    generate_pseudobulk(
        dense_adata,
        "cell_type",
        n_pseudobulks=4,
        target_proportion_min=1.0,
        target_proportion_max=1.0,
        n_cells_per_pseudobulk=2,
        random_state=11,
    )

    replacement_warnings = [
        warning
        for warning in recwarn
        if "replacement" in str(warning.message).lower()
    ]
    assert replacement_warnings == []
