from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk_specific import align_specific_to_universe

def test_align_specific_to_universe_basic():
    """Test standard alignment where universe overlaps with the dataframe."""
    df = pd.DataFrame(
        {
            "2023-01-01": [0.1, 0.2, 0.3],
            "2023-01-02": [0.4, 0.5, 0.6],
        },
        index=["AAPL", "MSFT", "GOOGL"]
    )
    universe = pd.Index(["AAPL", "GOOGL", "TSLA"])

    result = align_specific_to_universe(df, universe)

    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == ["AAPL", "GOOGL", "TSLA"]
    assert list(result.columns) == ["2023-01-01", "2023-01-02"]

    # AAPL and GOOGL should keep their values, MSFT should be dropped
    assert result.loc["AAPL", "2023-01-01"] == 0.1
    assert result.loc["GOOGL", "2023-01-02"] == 0.6

    # TSLA should be filled with 0.0 (default fill_value)
    assert result.loc["TSLA", "2023-01-01"] == 0.0
    assert result.loc["TSLA", "2023-01-02"] == 0.0

def test_align_specific_to_universe_custom_fill():
    """Test alignment with a custom fill value."""
    df = pd.DataFrame(
        {"2023-01-01": [1.0, 2.0]},
        index=["A", "B"]
    )
    universe = pd.Index(["A", "C"])

    result = align_specific_to_universe(df, universe, fill_value=np.nan)

    assert result.loc["A", "2023-01-01"] == 1.0
    assert np.isnan(result.loc["C", "2023-01-01"])

def test_align_specific_to_universe_empty_input():
    """Test alignment when input dataframe is empty."""
    df = pd.DataFrame()
    universe = pd.Index(["A", "B"])

    result = align_specific_to_universe(df, universe)

    # In Pandas, reindexing an empty dataframe with no columns results in a dataframe with the new index and no columns
    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == ["A", "B"]
    assert len(result.columns) == 0

def test_align_specific_to_universe_empty_universe():
    """Test alignment when the stock universe is empty."""
    df = pd.DataFrame(
        {"2023-01-01": [1.0, 2.0]},
        index=["A", "B"]
    )
    universe = pd.Index([])

    result = align_specific_to_universe(df, universe)

    assert isinstance(result, pd.DataFrame)
    assert len(result.index) == 0
    assert list(result.columns) == ["2023-01-01"]
