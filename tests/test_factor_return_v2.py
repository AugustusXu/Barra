import numpy as np
import pandas as pd
import pytest

from src.factor_return_v2 import _winsorize_and_standardize

def test_winsorize_and_standardize_normal():
    np.random.seed(42)
    # Generate some roughly normal data
    data = pd.Series(np.random.randn(100))
    result = _winsorize_and_standardize(data)

    # Should be standardized (mean roughly 0, std roughly 1)
    assert np.isclose(result.mean(), 0, atol=1e-5)
    assert np.isclose(result.std(), 1, atol=1e-5)
    # No NaNs expected
    assert not result.isna().any()

def test_winsorize_and_standardize_with_outliers():
    # Regular data
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    # Add huge outliers
    data_with_outliers = pd.Series(data + [1000.0, -1000.0])

    result = _winsorize_and_standardize(data_with_outliers)

    # Check that outliers are brought within bounds
    # original std before clipping is huge, after clipping should be much smaller.
    # We can check max/min values of the result
    assert result.max() < 10
    assert result.min() > -10

    # Result must be standardized
    assert np.isclose(result.mean(), 0, atol=1e-5)
    assert np.isclose(result.std(), 1, atol=1e-5)

def test_winsorize_and_standardize_with_nans():
    data = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    result = _winsorize_and_standardize(data)

    # Output should have NaN in the same place
    assert pd.isna(result[2])

    # Dropping NaN, it should be standardized
    valid_result = result.dropna()
    assert np.isclose(valid_result.mean(), 0, atol=1e-5)
    assert np.isclose(valid_result.std(), 1, atol=1e-5)

def test_winsorize_and_standardize_zero_mad_and_std():
    # Data where all values are identical
    data = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
    result = _winsorize_and_standardize(data)

    # Since std=0 fallback sets std=1.0 and mean=5.0, result should be exactly 0
    assert (result == 0.0).all()

def test_winsorize_and_standardize_custom_mad_k():
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

    # Larger mad_k means less clipping, leading to larger std and smaller standardized values for normal elements
    result_tight = _winsorize_and_standardize(data, mad_k=1.0)
    result_loose = _winsorize_and_standardize(data, mad_k=10.0)

    # The max value in loose clipping should be higher than in tight clipping
    # before standardization, but let's just check the values differ.
    assert not result_tight.equals(result_loose)

def test_winsorize_and_standardize_string_coerce():
    # Function converts using pd.to_numeric(errors="coerce")
    data = pd.Series([1, 2, "3", "invalid", 5])
    result = _winsorize_and_standardize(data)

    # "invalid" becomes NaN
    assert pd.isna(result[3])

    valid_result = result.dropna()
    assert len(valid_result) == 4
    assert np.isclose(valid_result.mean(), 0, atol=1e-5)
    assert np.isclose(valid_result.std(), 1, atol=1e-5)
