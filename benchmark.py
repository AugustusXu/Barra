import pandas as pd
import numpy as np
import time
from src.factor_eval import calc_group_nav

# Generate some dummy data
np.random.seed(42)
dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
n_stocks = 3000
n_days = len(dates)

data = {
    "trade_date": np.repeat(dates, n_stocks),
    "stock_id": np.tile(np.arange(n_stocks), n_days),
    "factor_val": np.random.randn(n_days * n_stocks),
    "pctchange": np.random.randn(n_days * n_stocks) * 2
}
df = pd.DataFrame(data)

start_time = time.time()
nav = calc_group_nav(df, factor_col="factor_val", return_col="pctchange", n_groups=5)
end_time = time.time()

print(f"Time taken: {end_time - start_time:.4f} seconds")
