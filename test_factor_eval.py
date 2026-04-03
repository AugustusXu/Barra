import pandas as pd
import numpy as np
from src.factor_eval import calc_group_nav

def test_calc_group_nav():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    n_stocks = 50
    n_days = len(dates)

    data = {
        "trade_date": np.repeat(dates, n_stocks),
        "stock_id": np.tile(np.arange(n_stocks), n_days),
        "factor_val": np.random.randn(n_days * n_stocks),
        "pctchange": np.random.randn(n_days * n_stocks) * 2
    }
    df = pd.DataFrame(data)

    nav = calc_group_nav(df, factor_col="factor_val", return_col="pctchange", n_groups=5)

    assert "G1" in nav
    assert "G5" in nav
    assert "LongShort" in nav

    # Check shape
    assert len(nav["G1"]) == n_days

test_calc_group_nav()
print("Tests passed!")
