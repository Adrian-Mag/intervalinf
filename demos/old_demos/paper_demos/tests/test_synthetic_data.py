"""Tests for synthetic-data target-fit helpers."""

import os
import sys

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../utils"))

from synthetic_data import _solve_target_smooth_alpha  # noqa: E402


def test_target_smooth_alpha_selects_largest_feasible_lambda():
    """Choose the smoothest candidate whose RMS standardized residual fits."""
    gram = np.eye(2)
    data = np.ones(2)
    data_std = np.ones(2)
    lambdas = np.array([0.1, 1.0, 10.0])

    result = _solve_target_smooth_alpha(
        gram, data, data_std, target_chi_rms=0.5, lambdas=lambdas,
    )

    assert result.target_met
    np.testing.assert_allclose(result.lambda_value, 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.chi_rms, 0.5, rtol=1e-12, atol=1e-12)


def test_target_smooth_alpha_reports_unreachable_target():
    """Return the best fit when the target is impossible."""
    gram = np.array([[1.0, 0.0], [0.0, 0.0]])
    data = np.array([0.0, 1.0])
    data_std = np.ones(2)
    lambdas = np.array([0.1, 1.0, 10.0])

    result = _solve_target_smooth_alpha(
        gram, data, data_std, target_chi_rms=0.5, lambdas=lambdas,
    )

    assert not result.target_met
    np.testing.assert_allclose(
        result.chi_rms, 1.0 / np.sqrt(2.0), rtol=1e-12, atol=1e-12,
    )
