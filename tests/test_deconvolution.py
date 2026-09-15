from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="deconvolution currently imports torch eagerly")

from deconversation.deconvolution import run_deconv


def test_nnls_recovers_known_mixture(signature_df, bulk_df):
    result = run_deconv(bulk_df, signature_df, solver="nnls")

    assert result.index.tolist() == ["sample_1"]
    assert result.columns.tolist() == ["type_a", "type_b"]
    np.testing.assert_allclose(result.loc["sample_1"], [0.7, 0.3], atol=1e-8)


def test_nnls_aligns_features_by_label(signature_df, bulk_df):
    reordered = bulk_df.loc[["gene_3", "gene_1", "gene_2"]]
    result = run_deconv(reordered, signature_df, solver="nnls")

    np.testing.assert_allclose(result.loc["sample_1"], [0.7, 0.3], atol=1e-8)


def test_nnls_normalizes_each_sample(signature_df):
    bulk = pd.DataFrame(
        {
            "sample_1": [7.0, 3.0, 10.0],
            "sample_2": [1.0, 4.0, 5.0],
        },
        index=signature_df.index,
    )
    result = run_deconv(bulk, signature_df, solver="nnls", normalize=True)

    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(result.loc["sample_1"], [0.7, 0.3], atol=1e-8)
    np.testing.assert_allclose(result.loc["sample_2"], [0.2, 0.8], atol=1e-8)


def test_no_shared_features_raises(signature_df):
    bulk = pd.DataFrame({"sample_1": [1.0]}, index=["other_gene"])

    with pytest.raises(ValueError, match="No common"):
        run_deconv(bulk, signature_df)


@pytest.mark.parametrize("solver", ["simplex", "simplex_nnls", "centered_simplex"])
def test_simplex_solvers_return_valid_proportions(solver, signature_df, bulk_df):
    result = run_deconv(bulk_df, signature_df, solver=solver)

    assert np.isfinite(result.to_numpy()).all()
    assert (result.to_numpy() >= -1e-10).all()
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-8)


def test_unknown_solver_raises_value_error(signature_df, bulk_df):
    with pytest.raises(ValueError, match="solver"):
        run_deconv(bulk_df, signature_df, solver="not_a_solver")


def test_dwls_never_returns_nan_for_zero_solution(signature_df):
    bulk = pd.DataFrame(
        {"sample_1": [-1.0, -1.0, -1.0]},
        index=signature_df.index,
    )
    result = run_deconv(bulk, signature_df, solver="dwls")

    assert np.isfinite(result.to_numpy()).all()


def test_gradient_solver_returns_valid_cpu_array(signature_df, bulk_df):
    result = run_deconv(bulk_df, signature_df, solver="gradient_descent")

    assert isinstance(result, pd.DataFrame)
    assert np.isfinite(result.to_numpy()).all()
    assert (result.to_numpy() >= 0).all()
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-5)
