from __future__ import annotations

import numpy as np
import pandas as pd


def to_specific_returns_wide(
    specific_returns_long: pd.DataFrame,
    stock_col: str = "stock_code",
    date_col: str = "trade_date",
    value_col: str = "specific_return",
) -> pd.DataFrame:
    """将长表特异性收益转为宽表（index=stock, columns=date）。"""
    req = {stock_col, date_col, value_col}
    if not req.issubset(specific_returns_long.columns):
        raise ValueError(f"specific_returns_long 缺少字段: {sorted(req - set(specific_returns_long.columns))}")

    data = specific_returns_long[[stock_col, date_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])
    wide = data.pivot_table(index=stock_col, columns=date_col, values=value_col, aggfunc="mean")
    wide = wide.sort_index().sort_index(axis=1)
    return wide


def align_specific_to_universe(
    specific_returns_wide: pd.DataFrame,
    stock_universe: pd.Index,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """按当期股票池顺序重排特异收益宽表。"""
    aligned = specific_returns_wide.reindex(stock_universe)
    return aligned.fillna(fill_value)


def _exp_weights(length: int, half_life: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=float)
    lam = 0.5 ** (1.0 / max(half_life, 1))
    idx = np.arange(length)
    raw = lam ** (length - 1 - idx)
    s = raw.sum()
    if s <= 0:
        return np.ones(length, dtype=float) / length
    return raw / s


def compute_specific_variance_matrix(
    specific_returns_wide: pd.DataFrame,
    target_date: pd.Timestamp | str,
    h: int = 252,
    tau: int = 90,
    lag: int = 5,
    short_half_life: int = 21,
) -> pd.DataFrame:
    """
    计算单日特异性方差矩阵（对角阵）：NW + 结构化收缩 + VRA。
    输入为 index=stock_code, columns=trade_date 的宽表。
    """
    if specific_returns_wide.empty:
        raise ValueError("specific_returns_wide 为空")

    y = specific_returns_wide.copy()
    y.columns = pd.to_datetime(y.columns)
    y = y.sort_index(axis=1)

    tgt = pd.to_datetime(target_date)
    if tgt not in y.columns:
        raise ValueError(f"target_date 不在特异收益列中: {tgt}")

    pos = y.columns.get_loc(tgt)
    if isinstance(pos, slice):
        pos = pos.stop - 1

    start_pos = int(pos) - h + 1
    if start_pos < 0:
        raise ValueError("历史窗口不足，无法计算特异方差")

    x_temp = y.iloc[:, start_pos : int(pos) + 1].to_numpy(dtype=float)
    stocks = y.index

    valid_mask = ~np.isnan(x_temp)
    x = np.nan_to_num(x_temp, nan=0.0)
    t1 = x.shape[1]

    w0 = _exp_weights(t1, tau)
    w0_mat = valid_mask * w0
    w0_sum = w0_mat.sum(axis=1, keepdims=True)
    w0_norm = np.divide(w0_mat, w0_sum, out=np.zeros_like(w0_mat, dtype=float), where=w0_sum != 0)

    x_bar = np.sum(x * w0_norm, axis=1, keepdims=True)
    gamma0 = np.sum(w0_norm * (x - x_bar) ** 2, axis=1)

    var_nw = gamma0.copy()

    for k in range(1, lag + 1):
        t2 = t1 - k
        if t2 <= 1:
            break

        wk = _exp_weights(t2, tau)
        x_t = x[:, :t2]
        x_l = x[:, k:]
        mask = valid_mask[:, :t2] & valid_mask[:, k:]

        wk_mat = mask * wk
        wk_sum = wk_mat.sum(axis=1, keepdims=True)
        wk_norm = np.divide(wk_mat, wk_sum, out=np.zeros_like(wk_mat, dtype=float), where=wk_sum != 0)

        x_t_bar = np.sum(x_t * wk_norm, axis=1, keepdims=True)
        x_l_bar = np.sum(x_l * wk_norm, axis=1, keepdims=True)
        gamma_diag = np.sum(wk_norm * (x_t - x_t_bar) * (x_l - x_l_bar), axis=1)

        bartlett = 1.0 - k / (lag + 1.0)
        var_nw += bartlett * 2.0 * gamma_diag

    var_nw = 21.0 * var_nw

    valid_vars = var_nw[var_nw > 0]
    if len(valid_vars) > 0:
        sigma_struct = float(np.median(valid_vars))
        shrink_v = 0.5
        var_nw = shrink_v * var_nw + (1.0 - shrink_v) * sigma_struct

    ws = _exp_weights(t1, short_half_life)
    ws_mat = valid_mask * ws
    ws_sum = ws_mat.sum(axis=1, keepdims=True)
    ws_norm = np.divide(ws_mat, ws_sum, out=np.zeros_like(ws_mat, dtype=float), where=ws_sum != 0)

    x_bar_s = np.sum(x * ws_norm, axis=1, keepdims=True)
    var_short = 21.0 * np.sum(ws_norm * (x - x_bar_s) ** 2, axis=1)

    valid_vra = (var_nw > 0) & (var_short > 0)
    if np.any(valid_vra):
        vra = float(np.median(var_short[valid_vra] / var_nw[valid_vra]))
        vra = float(np.clip(vra, 0.5, 2.5))
        var_nw = var_nw * vra

    var_nw = np.clip(var_nw, a_min=1e-10, a_max=None)
    delta = np.diag(var_nw)
    return pd.DataFrame(delta, index=stocks, columns=stocks)
