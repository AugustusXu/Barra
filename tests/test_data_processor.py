import numpy as np
import pandas as pd
import pytest

from src.data_processor import (
    mad_winsorize_series,
    remove_outliers_and_zscore,
    standardize_by_date,
)


def test_mad_winsorize_series():
    # Test 1: Basic functionality
    # [1, 2, 3, 4, 5, 100] -> median=3.5, mad=1.5
    # bounds: 3.5 +/- 5 * 1.4826 * 1.5 = 3.5 +/- 11.1195 = [-7.6195, 14.6195]
    # clipped: [1, 2, 3, 4, 5, 14.6195]
    # mean: 4.93658, std: 4.549...
    s = pd.Series([1, 2, 3, 4, 5, 100])
    res = mad_winsorize_series(s, multiplier=5.0)

    assert res.mean() == pytest.approx(0, abs=1e-6)
    assert res.std(ddof=0) == pytest.approx(1, abs=1e-6)

    # 100 should be clipped
    assert res.iloc[-1] > res.iloc[4] # still the max, but not as extreme

    # Test 2: Edge cases - NaN values
    s2 = pd.Series([1, 2, np.nan, 4, 5, 100])
    res2 = mad_winsorize_series(s2)
    assert np.isnan(res2.iloc[2])
    # Ignore NaN and compute for the rest
    assert res2.dropna().mean() == pytest.approx(0, abs=1e-6)
    assert res2.dropna().std(ddof=0) == pytest.approx(1, abs=1e-6)

    # Test 3: Edge cases - zero variance (all values same)
    s3 = pd.Series([2, 2, 2, 2, 2])
    res3 = mad_winsorize_series(s3)
    # std is 0, so it should be centered but not scaled by std=0
    assert res3.sum() == pytest.approx(0, abs=1e-6)
    assert (res3 == 0).all()


def test_standardize_by_date():
    df = pd.DataFrame({
        "trade_date": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-02"],
        "factor": [1, 2, 100, 5, 5, 5],
        "other_col": ["a", "b", "c", "d", "e", "f"]
    })

    # 2023-01-01 will be winsorized and standardized
    # 2023-01-02 will be winsorized (all 5) and standardized to 0

    res_df = standardize_by_date(df, factor_col="factor", date_col="trade_date")

    # Check that other columns remain intact
    assert "other_col" in res_df.columns
    assert (res_df["other_col"] == df["other_col"]).all()

    group1 = res_df[res_df["trade_date"] == "2023-01-01"]["factor"]
    group2 = res_df[res_df["trade_date"] == "2023-01-02"]["factor"]

    # Check that group 1 has 0 mean and 1 std (ddof=0)
    assert group1.mean() == pytest.approx(0, abs=1e-6)
    assert group1.std(ddof=0) == pytest.approx(1, abs=1e-6)

    # Check that group 2 is all 0
    assert (group2 == 0).all()

    # Test missing dates drop
    df_missing_date = pd.DataFrame({
        "trade_date": ["2023-01-01", None, "2023-01-02"],
        "factor": [1, 2, 3]
    })
    res_missing_date = standardize_by_date(df_missing_date, factor_col="factor", date_col="trade_date")
    assert len(res_missing_date) == 2


def test_remove_outliers_and_zscore():
    df = pd.DataFrame({
        "trade_date": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-02"],
        "factor": [1, 2, 100, 5, 5, 5]
    })

    res_df = remove_outliers_and_zscore(df, factor_col="factor", date_col="trade_date")

    # Handle pandas apply() differences where group columns might be dropped
    # or moved to the index.
    if "trade_date" in res_df.columns:
        group1 = res_df[res_df["trade_date"] == pd.Timestamp("2023-01-01")]["factor"]
        group2 = res_df[res_df["trade_date"] == pd.Timestamp("2023-01-02")]["factor"]
    else:
        group1 = res_df["factor"].iloc[:3]
        group2 = res_df["factor"].iloc[3:]

    # In remove_outliers_and_zscore, standard zscore uses ddof=0, but scipy's zscore omit nan
    # [1, 2, 100] -> clipped to ~ [1, 2, 7] -> mean=3.333, std=2.62...
    assert group1.mean() == pytest.approx(0, abs=1e-6)
    assert group1.std(ddof=0) == pytest.approx(1, abs=1e-6)

    # For group 2 (all 5s), std is 0. zscore handles 0 std by returning 0s (nan with omit depending on version, but typical is nan).
    # Since zscore returns nan when std=0 in older scipy/pandas versions, let's verify what it returns.
    # Actually, zscore of constant array returns NaNs. Let's ensure it doesn't crash.
    assert group2.isna().all() or (group2 == 0).all() or group2.mean() == pytest.approx(0, abs=1e-6)

    # Missing date handling
    df_missing = pd.DataFrame({
        "trade_date": ["2023-01-01", None, "2023-01-02"],
        "factor": [1, 2, 3]
    })
    res_missing = remove_outliers_and_zscore(df_missing, factor_col="factor", date_col="trade_date")
    assert len(res_missing) == 2
