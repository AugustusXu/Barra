import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(42)
X = np.random.randn(100, 3)
y = np.random.randn(100)
w = np.random.rand(100) * 100
w_sqrt = np.sqrt(w)

model = sm.WLS(y, X, weights=w_sqrt)
result = model.fit()

# Native approach
sqrt_w_for_scaling = np.sqrt(w_sqrt)
x_w = X * sqrt_w_for_scaling[:, None]
y_w = y * sqrt_w_for_scaling
beta = np.linalg.pinv(x_w.T @ x_w) @ (x_w.T @ y_w)

print("sm params:", result.params)
print("np params:", beta)

# Residuals
pred = X @ beta
resid = y - pred

print("sm resid max diff:", np.max(np.abs(result.resid - resid)))
