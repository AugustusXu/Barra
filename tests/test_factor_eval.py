import pandas as pd
import numpy as np
from unittest.mock import patch
from src.factor_eval import calc_group_nav

def test_calc_group_nav_qcut_value_error():
    """
    Test that calc_group_nav correctly catches ValueError from pd.qcut
    and continues gracefully, skipping the date group.
    """
    df = pd.DataFrame({
        "trade_date": ["2023-01-01"] * 10,
        "factor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "pctchange": np.random.randn(10)
    })

    # Mock pd.qcut to force it to raise ValueError
    with patch('pandas.qcut', side_effect=ValueError("Mocked ValueError")):
        nav = calc_group_nav(df, factor_col="factor", return_col="pctchange", n_groups=5)

    # Assert that no groups or valid long_short returns were recorded due to the error
    assert isinstance(nav, dict)
    assert "G1" in nav
    assert "LongShort" in nav
    assert len(nav["G1"]) == 0
    assert len(nav["LongShort"]) == 0

def test_calc_group_nav_constant_values():
    """
    Test that calc_group_nav skips groups when pd.qcut generates fewer bins
    due to constant values and duplicates='drop'.
    """
    df = pd.DataFrame({
        "trade_date": ["2023-01-01"] * 10,
        "factor": [1.0] * 10, # Constant values to trigger drop
        "pctchange": np.random.randn(10)
    })

    nav = calc_group_nav(df, factor_col="factor", return_col="pctchange", n_groups=5)

    # Assert that no groups or valid long_short returns were recorded due to too few bins
    assert isinstance(nav, dict)
    assert "G1" in nav
    assert "LongShort" in nav
    assert len(nav["G1"]) == 0
    assert len(nav["LongShort"]) == 0

def test_calc_group_nav_happy_path():
    """
    Test calc_group_nav computes NAVs correctly under normal conditions.
    """
    # Create 2 days of valid data with 10 items each
    np.random.seed(42)
    dates = ["2023-01-01"] * 10 + ["2023-01-02"] * 10
    factors = list(range(10)) + list(range(10))
    pctchanges = np.random.randn(20) / 100  # small pct changes

    df = pd.DataFrame({
        "trade_date": dates,
        "factor": factors,
        "pctchange": pctchanges
    })

    nav = calc_group_nav(df, factor_col="factor", return_col="pctchange", n_groups=5)

    assert isinstance(nav, dict)
    assert "G1" in nav
    assert "G5" in nav
    assert "LongShort" in nav

    # For 2 dates, we expect nav to have length 2
    assert len(nav["G1"]) == 2
    assert len(nav["G5"]) == 2
    assert len(nav["LongShort"]) == 2

    # The return index should be timestamps
    assert isinstance(nav["G1"].index[0], pd.Timestamp)
    assert nav["G1"].index[0] == pd.Timestamp("2023-01-01")
    assert nav["G1"].index[1] == pd.Timestamp("2023-01-02")
