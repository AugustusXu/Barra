import numpy as np
import pandas as pd
import pytest

from src.data_processor import mad_winsorize_series

def test_mad_winsorize_series_normal():
    # Construct a series with known median and mad, and add outliers
    # base values: 1, 2, 3, 4, 5 -> median = 3, mad = 1 (abs dev: 2, 1, 0, 1, 2)
    s = pd.Series([1, 2, 3, 4, 5, 100, -100])

    res = mad_winsorize_series(s, multiplier=2.0)

    # 1, 2, 3, 4, 5, 100, -100
    # Median is 3. Abs dev: 2, 1, 0, 1, 2, 97, 103
    # Sorted abs dev: 0, 1, 1, 2, 2, 97, 103 -> median abs dev (MAD) is 2.0
    # multiplier = 2.0
    # lower = 3 - 2 * 1.4826 * 2.0 = 3 - 5.9304 = -2.9304
    # upper = 3 + 2 * 1.4826 * 2.0 = 3 + 5.9304 = 8.9304
    # Clipped series: [1, 2, 3, 4, 5, 8.9304, -2.9304]

    clipped_mean = np.mean([1, 2, 3, 4, 5, 8.9304, -2.9304])
    clipped_std = np.std([1, 2, 3, 4, 5, 8.9304, -2.9304])

    # Check if the result matches expected normalization
    expected = (np.array([1, 2, 3, 4, 5, 8.9304, -2.9304]) - clipped_mean) / clipped_std

    np.testing.assert_allclose(res.to_numpy(), expected)

def test_mad_winsorize_series_with_nans():
    # Series with NaNs and unconvertible strings
    s = pd.Series([1, 2, np.nan, 4, 5, '100', 'a'])

    res = mad_winsorize_series(s)

    # Valid numeric: 1.0, 2.0, NaN, 4.0, 5.0, 100.0, NaN
    # Valid elements: 1, 2, 4, 5, 100 -> median = 4.0
    # Abs dev: 3, 2, 0, 1, 96 -> sorted: 0, 1, 2, 3, 96 -> MAD = 2.0
    # lower = 4.0 - 5 * 1.4826 * 2.0 = -10.826
    # upper = 4.0 + 5 * 1.4826 * 2.0 = 18.826
    # Clipped valid: 1, 2, 4, 5, 18.826

    clipped = np.array([1, 2, 4, 5, 18.826])
    clipped_mean = np.mean(clipped)
    clipped_std = np.std(clipped)

    expected_valid = (clipped - clipped_mean) / clipped_std

    # res should have same index as s, with NaNs at index 2 and 6
    assert pd.isna(res.iloc[2])
    assert pd.isna(res.iloc[6])

    np.testing.assert_allclose(res.dropna().to_numpy(), expected_valid)

def test_mad_winsorize_series_mad_zero():
    # All same values will result in MAD = 0
    s = pd.Series([5, 5, 5, 5, 5])

    res = mad_winsorize_series(s)

    # clipped will be 5, 5, 5, 5, 5
    # std will be 0
    # when std=0, returns clipped - mean -> 0, 0, 0, 0, 0
    np.testing.assert_allclose(res.to_numpy(), [0, 0, 0, 0, 0])

def test_mad_winsorize_series_all_nans():
    # Series with all NaNs
    s = pd.Series([np.nan, np.nan, np.nan])

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mad_winsorize_series(s)

    assert res.isna().all()
    assert len(res) == 3

def test_mad_winsorize_series_preserve_index():
    s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
    res = mad_winsorize_series(s)

    np.testing.assert_array_equal(res.index, s.index)
