import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.data_loader import load_pctchange_daily_pkl


def test_load_pctchange_daily_pkl_unpickling_error():
    """
    Test that load_pctchange_daily_pkl gracefully handles unpickling errors
    by skipping the problematic file and continuing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create dummy pkl files representing trading days
        file1 = tmp_path / "2023-01-01.pkl"
        file2 = tmp_path / "2023-01-02.pkl"
        file3 = tmp_path / "2023-01-03.pkl"

        # Touch files so they exist when globbing
        file1.touch()
        file2.touch()
        file3.touch()

        # Valid data
        df1 = pd.DataFrame({
            "stock_code": ["000001.SZ"],
            "trade_date": ["2023-01-01"],
            "pctchange": [0.01]
        })

        df3 = pd.DataFrame({
            "stock_code": ["000002.SZ"],
            "trade_date": ["2023-01-03"],
            "pctchange": [0.03]
        })

        def mock_read_pickle(filepath):
            filepath_str = str(filepath)
            if "2023-01-02" in filepath_str:
                raise Exception("Simulated unpickling error")
            elif "2023-01-01" in filepath_str:
                return df1
            elif "2023-01-03" in filepath_str:
                return df3
            return pd.DataFrame()

        with patch("pandas.read_pickle", side_effect=mock_read_pickle):
            result = load_pctchange_daily_pkl(tmp_path)

            # The result should contain data from df1 and df3, skipping the error from df2
            assert len(result) == 2
            assert result["stock_code"].tolist() == ["000001.SZ", "000002.SZ"]
            assert result["pctchange"].tolist() == [0.01, 0.03]
