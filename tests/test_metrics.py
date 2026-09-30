import math

import pytest

from llm_numeric_inputs.experiments.metrics import regression_metrics


def test_regression_metrics_known_values():
    metrics = regression_metrics([1.0, 3.0], [2.0, 1.0], (0.0, 10.0))
    assert metrics["count"] == 2
    assert metrics["mse"] == pytest.approx(2.5)
    assert metrics["rmse"] == pytest.approx(math.sqrt(2.5))
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["normalized_mae_over_range"] == pytest.approx(0.15)
    assert metrics["max_abs_error"] == pytest.approx(2.0)


def test_regression_metrics_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        regression_metrics([], [])


def test_regression_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same count"):
        regression_metrics([1.0], [1.0, 2.0])
