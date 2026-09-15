from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deconversation.evaluation import compute_correlation, compute_rmse


@pytest.fixture
def metric_frames():
    true = pd.DataFrame(
        {"type_a": [0.8, 0.2], "type_b": [0.2, 0.8]},
        index=["sample_1", "sample_2"],
    )
    predicted = pd.DataFrame(
        {"type_a": [0.7, 0.3], "type_b": [0.3, 0.7]},
        index=true.index,
    )
    return true, predicted


def test_rmse_matches_hand_calculation(metric_frames):
    true, predicted = metric_frames
    assert compute_rmse(true, predicted) == pytest.approx(0.1)


def test_global_pearson_matches_numpy(metric_frames):
    true, predicted = metric_frames
    expected = np.corrcoef(true.to_numpy().ravel(), predicted.to_numpy().ravel())[0, 1]
    assert compute_correlation(true, predicted) == pytest.approx(expected)


def test_metrics_align_reordered_labels(metric_frames):
    true, predicted = metric_frames
    reordered = predicted.loc[["sample_2", "sample_1"], ["type_b", "type_a"]]

    assert compute_rmse(true, reordered) == pytest.approx(compute_rmse(true, predicted))
    assert compute_correlation(true, reordered) == pytest.approx(
        compute_correlation(true, predicted)
    )


def test_rmse_returns_requested_breakdowns(metric_frames):
    true, predicted = metric_frames
    result = compute_rmse(
        true,
        predicted,
        return_per_sample=True,
        return_per_celltype=True,
    )

    assert set(result) == {"overall", "per_sample", "per_celltype"}
    np.testing.assert_allclose(result["per_sample"], 0.1)
    np.testing.assert_allclose(result["per_celltype"], 0.1)


def test_correlation_returns_requested_breakdowns(metric_frames):
    true, predicted = metric_frames
    result = compute_correlation(
        true,
        predicted,
        return_per_sample=True,
        return_per_celltype=True,
    )

    assert set(result) == {"overall", "per_sample", "per_celltype"}
    assert result["per_sample"].index.tolist() == true.index.tolist()
    assert result["per_celltype"].index.tolist() == true.columns.tolist()


@pytest.mark.parametrize("function", [compute_rmse, compute_correlation])
def test_metrics_reject_completely_disjoint_labels(function, metric_frames):
    true, predicted = metric_frames
    predicted.index = ["other_1", "other_2"]

    with pytest.raises(ValueError, match="No overlapping"):
        function(true, predicted)
