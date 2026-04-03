import time
import numpy as np
import pandas as pd
from src.factor_return import compute_daily_factor_returns

# Generate dummy data
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=100, freq="D")
stocks = [f"stock_{i}" for i in range(500)]

data = []
for d in dates:
    for s in stocks:
        data.append({
            "trade_date": d,
            "stock_code": s,
            "factor_1": np.random.randn(),
            "factor_2": np.random.randn(),
            "weight": np.random.rand() * 100,
            "next_return": np.random.randn() * 0.05
        })

df = pd.DataFrame(data)

start = time.time()
factor_returns, specific_returns = compute_daily_factor_returns(
    panel_df=df,
    factor_cols=["factor_1", "factor_2"],
    weight_col="weight",
    min_stocks=30
)
end = time.time()
print(f"Time taken: {end - start:.4f} seconds")
