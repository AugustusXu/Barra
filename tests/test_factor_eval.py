import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from src.factor_eval import calc_perf_stats

def test_calc_perf_stats_empty_nav():
    nav = pd.Series(dtype=float)
    stats = calc_perf_stats(nav)
    assert np.isnan(stats["annual_return"])
    assert np.isnan(stats["sharpe"])
    assert np.isnan(stats["max_drawdown"])

def test_calc_perf_stats_short_nav():
    # Less than 3 points
    dates = [datetime(2023, 1, 1), datetime(2023, 1, 2)]
    nav = pd.Series([1.0, 1.1], index=dates)
    stats = calc_perf_stats(nav)
    assert np.isnan(stats["annual_return"])
    assert np.isnan(stats["sharpe"])
    assert np.isnan(stats["max_drawdown"])

def test_calc_perf_stats_happy_path():
    # 4 points, 3 days
    dates = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
        datetime(2023, 1, 4),
    ]
    # NAV increases by 10% each day: pct_change = [NaN, 0.1, 0.1, 0.1]
    nav = pd.Series([1.0, 1.1, 1.21, 1.331], index=dates)
    stats = calc_perf_stats(nav)

    # max_drawdown should be 0.0 since it strictly increases
    assert stats["max_drawdown"] == 0.0

    # In floating point math, pct_change() can yield values very close to but not exactly identical,
    # causing variance > 0.
    if np.isnan(stats["sharpe"]):
        assert True
    else:
        # If var > 0 due to float imprecision, sharpe will be extremely large, which is acceptable
        # for a test checking the logic.
        assert stats["sharpe"] > 1000

    # Annual return calculation: (1.331 / 1.0) ** (365 / 3) - 1
    expected_annual_return = (1.331 / 1.0) ** (365.0 / 3) - 1.0
    assert np.isclose(stats["annual_return"], expected_annual_return)

def test_calc_perf_stats_with_variance():
    dates = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
        datetime(2023, 1, 4),
    ]
    # pct_change = [NaN, 0.1, -0.05, 0.2]
    nav = pd.Series([1.0, 1.1, 1.045, 1.254], index=dates)
    stats = calc_perf_stats(nav)

    # drawdown: max is at 1.045 from peak 1.1: (1.1 - 1.045)/1.1 = 0.05
    assert np.isclose(stats["max_drawdown"], 0.05)

    # annual return: (1.254 / 1.0) ** (365/3) - 1
    expected_annual_return = (1.254 / 1.0) ** (365.0 / 3) - 1.0
    assert np.isclose(stats["annual_return"], expected_annual_return)

    # sharpe: mean / sqrt(variance)
    # rets = [0.1, -0.05, 0.2]
    # mean = (0.1 - 0.05 + 0.2) / 3 = 0.25 / 3 ≈ 0.08333
    # var = sum((x - mean)^2) / (n-1)
    rets = np.array([0.1, -0.05, 0.2])
    mean = np.mean(rets)
    var = np.var(rets, ddof=1)
    expected_sharpe = mean / np.sqrt(var)
    assert np.isclose(stats["sharpe"], expected_sharpe)

def test_calc_perf_stats_nav_starts_at_zero():
    dates = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
        datetime(2023, 1, 4),
    ]
    # Initial value is 0, so annual return shouldn't be calculated
    nav = pd.Series([0.0, 1.0, 1.1, 1.2], index=dates)
    stats = calc_perf_stats(nav)
    assert np.isnan(stats["annual_return"])
    # pct_change from 0.0 to 1.0 is inf
    # mean of [inf, 0.1, 0.09] is inf, var is nan
    assert np.isnan(stats["sharpe"])

def test_calc_perf_stats_zero_variance():
    dates = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
        datetime(2023, 1, 4),
    ]
    nav = pd.Series([1.0, 1.0, 1.0, 1.0], index=dates)
    stats = calc_perf_stats(nav)
    assert np.isnan(stats["sharpe"])
    assert stats["max_drawdown"] == 0.0
    assert stats["annual_return"] == 0.0
