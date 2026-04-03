import pytest
import numpy as np
import pandas as pd
from src.risk_specific import compute_specific_variance_matrix

def test_compute_specific_variance_matrix_empty():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="specific_returns_wide 为空"):
        compute_specific_variance_matrix(df, target_date="2023-01-01")

def test_compute_specific_variance_matrix_missing_target():
    dates = pd.date_range("2023-01-01", periods=3)
    df = pd.DataFrame(np.random.randn(2, 3), index=["A", "B"], columns=dates)
    with pytest.raises(ValueError, match="target_date 不在特异收益列中"):
        compute_specific_variance_matrix(df, target_date="2023-01-05", h=3)

def test_compute_specific_variance_matrix_insufficient_history():
    dates = pd.date_range("2023-01-01", periods=3)
    df = pd.DataFrame(np.random.randn(2, 3), index=["A", "B"], columns=dates)
    with pytest.raises(ValueError, match="历史窗口不足，无法计算特异方差"):
        compute_specific_variance_matrix(df, target_date="2023-01-03", h=5)

def test_compute_specific_variance_matrix_happy_path():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=10)
    df = pd.DataFrame(np.random.randn(3, 10), index=["S1", "S2", "S3"], columns=dates)

    # We need h <= 10. Let's use h=5, tau=90, lag=2, short_half_life=21
    res = compute_specific_variance_matrix(df, target_date="2023-01-10", h=5, tau=90, lag=2, short_half_life=21)

    assert isinstance(res, pd.DataFrame)
    assert list(res.index) == ["S1", "S2", "S3"]
    assert list(res.columns) == ["S1", "S2", "S3"]

    # Check if diagonal
    mat = res.to_numpy()
    off_diagonal = mat - np.diag(np.diag(mat))
    assert np.allclose(off_diagonal, 0.0)

    # Check values are positive
    assert np.all(np.diag(mat) > 0)

def test_compute_specific_variance_matrix_with_nan():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=10)
    data = np.random.randn(3, 10)
    data[0, 5] = np.nan  # Inject NaN
    df = pd.DataFrame(data, index=["S1", "S2", "S3"], columns=dates)

    res = compute_specific_variance_matrix(df, target_date="2023-01-10", h=5, tau=90, lag=2, short_half_life=21)
    assert isinstance(res, pd.DataFrame)
    mat = res.to_numpy()
    assert np.all(np.diag(mat) > 0)
    assert np.allclose(mat - np.diag(np.diag(mat)), 0.0)
