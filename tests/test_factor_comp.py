import pytest
import pandas as pd
from src.factor_comp import compute_volatility_factors

def test_compute_volatility_factors_missing_index():
    stock_price = pd.DataFrame({
        "stock_code": ["000001.SZ"],
        "trade_date": ["2020-01-01"],
        "close": [10.0],
        "preclose": [9.0],
        "pctchange": [11.11]
    })

    index_price = pd.DataFrame({
        "stock_code": ["000905.SH"], # Intentionally missing "000300.SH"
        "trade_date": ["2020-01-01"],
        "close": [1000.0],
        "preclose": [900.0],
        "pctchange": [11.11]
    })

    with pytest.raises(ValueError, match="波动率计算缺少市场指数列: 000300.SH"):
        compute_volatility_factors(stock_price, index_price, market_code="000300.SH")
