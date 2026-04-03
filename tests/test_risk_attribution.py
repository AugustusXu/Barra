import numpy as np
import pandas as pd
import pytest

from src.risk_attribution import _safe_scalar, attribute_portfolio_risk


def test_safe_scalar():
    # Test float
    assert _safe_scalar(5.0) == 5.0

    # Test 0d array
    assert _safe_scalar(np.array(3.14)) == 3.14

    # Test 1d array with 1 element
    assert _safe_scalar(np.array([2.718])) == 2.718

    # Test 2d array with 1 element
    assert _safe_scalar(np.array([[1.414]])) == 1.414

    # Test NaN
    assert _safe_scalar(np.nan) == 0.0
    assert _safe_scalar(np.array([np.nan])) == 0.0


def test_attribute_portfolio_risk():
    weights = pd.Series([0.4, 0.6], index=["AAPL", "MSFT"])
    benchmark_weights = pd.Series([0.5, 0.5], index=["AAPL", "MSFT"])

    exposure_matrix = pd.DataFrame(
        {
            "Size": [1.0, -0.5],
            "Ind_Tech": [1.0, 1.0]
        }, index=["AAPL", "MSFT"]
    )

    factor_cov = pd.DataFrame(
        {
            "Size": [0.04, 0.0],
            "Ind_Tech": [0.0, 0.01]
        }, index=["Size", "Ind_Tech"]
    )

    specific_var_diag = pd.DataFrame(
        {
            "AAPL": [0.05, 0.0],
            "MSFT": [0.0, 0.03]
        }, index=["AAPL", "MSFT"]
    )

    res = attribute_portfolio_risk(
        weights,
        exposure_matrix,
        factor_cov,
        specific_var_diag,
        benchmark_weights
    )

    assert "summary" in res
    assert "active_weights" in res

    # Check wa = w - wb = [-0.1, 0.1]
    wa = res["active_weights"]
    np.testing.assert_allclose(wa.values, [-0.1, 0.1])

    summary = res["summary"]
    assert hasattr(summary, "total_variance")
    assert hasattr(summary, "factor_variance")
    assert hasattr(summary, "specific_variance")
    assert hasattr(summary, "style_variance")
    assert hasattr(summary, "industry_variance")

    # Manual calculation:
    # wa = [-0.1, 0.1]
    # x = [[1, 1], [-0.5, 1]]
    # f = [[0.04, 0], [0, 0.01]]
    # d = diag(0.05, 0.03)
    # x.T @ wa = [-0.15, 0]
    # factor_var = b.T @ f @ b = [-0.15, 0] @ [[0.04, 0], [0, 0.01]] @ [-0.15, 0] = 0.0009
    # specific_var = wa.T @ d @ wa = 0.01 * 0.05 + 0.01 * 0.03 = 0.0008
    # total_var = 0.0017

    np.testing.assert_allclose(summary.factor_variance, 0.0009)
    np.testing.assert_allclose(summary.specific_variance, 0.0008)
    np.testing.assert_allclose(summary.total_variance, 0.0017)
    np.testing.assert_allclose(summary.style_variance, 0.0009)
    np.testing.assert_allclose(summary.industry_variance, 0.0)


def test_attribute_portfolio_risk_empty_weights():
    with pytest.raises(ValueError, match="weights 为空"):
        attribute_portfolio_risk(
            pd.Series(dtype=float),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame()
        )

def test_attribute_portfolio_risk_no_benchmark():
    weights = pd.Series([0.4, 0.6], index=["AAPL", "MSFT"])
    exposure_matrix = pd.DataFrame(
        {
            "Size": [1.0, -0.5],
            "Ind_Tech": [1.0, 1.0]
        }, index=["AAPL", "MSFT"]
    )

    factor_cov = pd.DataFrame(
        {
            "Size": [0.04, 0.0],
            "Ind_Tech": [0.0, 0.01]
        }, index=["Size", "Ind_Tech"]
    )

    specific_var_diag = pd.DataFrame(
        {
            "AAPL": [0.05, 0.0],
            "MSFT": [0.0, 0.03]
        }, index=["AAPL", "MSFT"]
    )

    res = attribute_portfolio_risk(
        weights,
        exposure_matrix,
        factor_cov,
        specific_var_diag
    )

    assert "summary" in res
    assert "active_weights" in res
    wa = res["active_weights"]
    np.testing.assert_allclose(wa.values, [0.4, 0.6])
