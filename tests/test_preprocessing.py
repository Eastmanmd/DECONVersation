from __future__ import annotations

import numpy as np
import pytest

from deconversation.preprocessing import create_signature_matrix


@pytest.mark.parametrize("adata_fixture", ["dense_adata", "sparse_adata"])
def test_create_signature_matrix_dense_and_sparse(request, adata_fixture):
    adata = request.getfixturevalue(adata_fixture)
    signature = create_signature_matrix(
        adata,
        sample_col="sample",
        cell_type_col="cell_type",
        groupby="cell_type",
    )

    assert signature.index.tolist() == ["gene_1", "gene_2", "gene_3"]
    assert signature.columns.tolist() == ["type_a", "type_b"]
    np.testing.assert_allclose(signature["type_a"], [3.0, 0.0, 2.0])
    np.testing.assert_allclose(signature["type_b"], [0.0, 4.0, 2.0])


def test_signature_matrix_can_subset_samples(dense_adata):
    signature = create_signature_matrix(
        dense_adata,
        sample_col="sample",
        cell_type_col="cell_type",
        groupby="cell_type",
        sample_ids=["donor_1"],
    )

    np.testing.assert_allclose(signature["type_a"], [2.0, 0.0, 1.0])
    np.testing.assert_allclose(signature["type_b"], [0.0, 6.0, 1.0])


def test_signature_matrix_can_subset_cell_types(dense_adata):
    signature = create_signature_matrix(
        dense_adata,
        sample_col="sample",
        cell_type_col="cell_type",
        groupby="cell_type",
        celltypes=["type_a"],
    )

    assert signature.columns.tolist() == ["type_a"]


def test_signature_matrix_rejects_missing_metadata(dense_adata):
    with pytest.raises(ValueError, match="missing"):
        create_signature_matrix(
            dense_adata,
            sample_col="missing",
            cell_type_col="cell_type",
            groupby="cell_type",
        )


def test_signature_matrix_rejects_empty_subset(dense_adata):
    with pytest.raises(ValueError, match="No cells remain"):
        create_signature_matrix(
            dense_adata,
            sample_col="sample",
            cell_type_col="cell_type",
            groupby="cell_type",
            celltypes=["not_present"],
        )


def test_signature_matrix_does_not_change_expression_values(dense_adata):
    before = dense_adata.X.copy()
    create_signature_matrix(
        dense_adata,
        sample_col="sample",
        cell_type_col="cell_type",
        groupby="cell_type",
    )

    np.testing.assert_array_equal(dense_adata.X, before)
