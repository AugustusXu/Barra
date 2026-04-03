from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factor_return import compute_daily_factor_returns

def test_compute_daily_factor_returns_ols():
    np.random.seed(42)
    # create sample data
    n_stocks = 50
    dates = pd.date_range("2023-01-01", periods=2)

    data = []
    for date in dates:
        for i in range(n_stocks):
            data.append({
                "trade_date": date,
                "stock_code": f"stock_{i}",
                "factor_1": np.random.randn(),
                "factor_2": np.random.randn(),
                "next_return": np.random.randn() * 0.05
            })

    df = pd.DataFrame(data)

    # Compute returns using OLS
    factor_returns, specific_returns = compute_daily_factor_returns(
        panel_df=df,
        factor_cols=["factor_1", "factor_2"],
        min_stocks=30
    )

    # Check outputs
    assert not factor_returns.empty
    assert "factor_1" in factor_returns.columns
    assert "factor_2" in factor_returns.columns
    assert len(factor_returns) == 2  # 2 days

    assert not specific_returns.empty
    assert "specific_return" in specific_returns.columns
    assert len(specific_returns) == n_stocks * 2
    assert "trade_date" in specific_returns.columns
    assert "stock_code" in specific_returns.columns


def test_compute_daily_factor_returns_wls():
    np.random.seed(42)
    n_stocks = 50
    dates = pd.date_range("2023-01-01", periods=2)

    data = []
    for date in dates:
        for i in range(n_stocks):
            data.append({
                "trade_date": date,
                "stock_code": f"stock_{i}",
                "factor_1": np.random.randn(),
                "factor_2": np.random.randn(),
                "weight": np.abs(np.random.randn()) * 1000,
                "next_return": np.random.randn() * 0.05
            })

    df = pd.DataFrame(data)

    # Compute returns using WLS
    factor_returns, specific_returns = compute_daily_factor_returns(
        panel_df=df,
        factor_cols=["factor_1", "factor_2"],
        weight_col="weight",
        min_stocks=30
    )

    # Check outputs
    assert not factor_returns.empty
    assert "factor_1" in factor_returns.columns
    assert "factor_2" in factor_returns.columns
    assert len(factor_returns) == 2  # 2 days

    assert not specific_returns.empty
    assert len(specific_returns) == n_stocks * 2


def test_compute_daily_factor_returns_missing_target():
    df = pd.DataFrame({
        "trade_date": pd.date_range("2023-01-01", periods=50),
        "stock_code": [f"stock_{i}" for i in range(50)],
        "factor_1": np.random.randn(50)
    })

    with pytest.raises(ValueError, match="panel_df 缺少 next_return 列"):
        compute_daily_factor_returns(
            panel_df=df,
            factor_cols=["factor_1"]
        )


def test_compute_daily_factor_returns_min_stocks_filtering():
    np.random.seed(42)
    # create sample data with fewer than 30 stocks per day
    n_stocks = 20
    dates = pd.date_range("2023-01-01", periods=2)

    data = []
    for date in dates:
        for i in range(n_stocks):
            data.append({
                "trade_date": date,
                "stock_code": f"stock_{i}",
                "factor_1": np.random.randn(),
                "next_return": np.random.randn() * 0.05
            })

    df = pd.DataFrame(data)

    # Compute returns
    factor_returns, specific_returns = compute_daily_factor_returns(
        panel_df=df,
        factor_cols=["factor_1"],
        min_stocks=30
    )

    # Because all days have < 30 stocks, the output should be empty
    assert factor_returns.empty
    assert specific_returns.empty


def test_compute_daily_factor_returns_missing_data_handling():
    np.random.seed(42)
    n_stocks = 50
    dates = pd.date_range("2023-01-01", periods=2)

    data = []
    for date in dates:
        for i in range(n_stocks):
            data.append({
                "trade_date": date,
                "stock_code": f"stock_{i}",
                "factor_1": np.random.randn(),
                "next_return": np.random.randn() * 0.05
            })

    df = pd.DataFrame(data)

    # Introduce NaNs to cause the number of valid stocks to drop below min_stocks on day 1
    # Drop valid stocks on day 1 to be < 30
    df.loc[(df["trade_date"] == dates[0]) & (df.index < 25), "factor_1"] = np.nan

    # Drop 1 stock on day 2 so it still has > 30 valid stocks
    df.loc[(df["trade_date"] == dates[1]) & (df.index == 50), "next_return"] = np.nan

    factor_returns, specific_returns = compute_daily_factor_returns(
        panel_df=df,
        factor_cols=["factor_1"],
        min_stocks=30
    )

    # Day 1 should be skipped, Day 2 should be computed
    assert len(factor_returns) == 1
    assert factor_returns.index[0] == dates[1]

    # specific_returns should contain 49 stocks for day 2
    assert len(specific_returns) == 49
