import pytest
import pandas as pd
import numpy as np

from src.risk_covariance import compute_factor_covariance_matrix

@pytest.fixture
def mock_factor_returns():
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=300, freq="B")
    factors = ["Factor_A", "Factor_B", "Factor_C"]
    data = np.random.normal(0, 0.01, size=(300, 3))
    return pd.DataFrame(data, index=dates, columns=factors)

def test_compute_factor_covariance_matrix_happy_path(mock_factor_returns):
    target_date = mock_factor_returns.index[-1]
    factor_order = ["Factor_A", "Factor_B", "Factor_C"]

    cov_matrix = compute_factor_covariance_matrix(
        factor_returns=mock_factor_returns,
        factor_order=factor_order,
        target_date=target_date,
        cov_days=252,
        mc=50, # Use smaller mc for faster tests
        random_seed=42
    )

    # Check return type
    assert isinstance(cov_matrix, pd.DataFrame)

    # Check shape
    assert cov_matrix.shape == (3, 3)

    # Check index and columns
    assert list(cov_matrix.index) == factor_order
    assert list(cov_matrix.columns) == factor_order

    # Check symmetry
    np.testing.assert_allclose(cov_matrix.values, cov_matrix.values.T, atol=1e-8)

    # Check positive semi-definite (eigenvalues >= 0)
    eigvals = np.linalg.eigvalsh(cov_matrix.values)
    assert np.all(eigvals >= -1e-10)

def test_compute_factor_covariance_matrix_empty_df():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="factor_returns 为空"):
        compute_factor_covariance_matrix(empty_df, ["Factor_A"], "2020-01-01")

def test_compute_factor_covariance_matrix_missing_columns(mock_factor_returns):
    with pytest.raises(ValueError, match="factor_returns 缺少因子列"):
        compute_factor_covariance_matrix(
            mock_factor_returns,
            ["Factor_A", "Factor_Missing"],
            mock_factor_returns.index[-1]
        )

def test_compute_factor_covariance_matrix_missing_target_date(mock_factor_returns):
    with pytest.raises(ValueError, match="target_date 不在 factor_returns 索引中"):
        compute_factor_covariance_matrix(
            mock_factor_returns,
            ["Factor_A"],
            "2019-01-01"
        )

def test_compute_factor_covariance_matrix_insufficient_history(mock_factor_returns):
    # Try to calculate with 252 days history but target date is at index 100
    target_date = mock_factor_returns.index[100]
    with pytest.raises(ValueError, match="历史窗口不足，无法计算协方差"):
        compute_factor_covariance_matrix(
            mock_factor_returns,
            ["Factor_A"],
            target_date,
            cov_days=252
        )

def test_compute_factor_covariance_matrix_deterministic(mock_factor_returns):
    target_date = mock_factor_returns.index[-1]
    factor_order = ["Factor_A", "Factor_B"]

    cov1 = compute_factor_covariance_matrix(
        factor_returns=mock_factor_returns,
        factor_order=factor_order,
        target_date=target_date,
        random_seed=123
    )

    cov2 = compute_factor_covariance_matrix(
        factor_returns=mock_factor_returns,
        factor_order=factor_order,
        target_date=target_date,
        random_seed=123
    )

    pd.testing.assert_frame_equal(cov1, cov2)

    cov3 = compute_factor_covariance_matrix(
        factor_returns=mock_factor_returns,
        factor_order=factor_order,
        target_date=target_date,
        random_seed=456
    )

    # Should be different with different seed, though potentially close
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(cov1, cov3)
