from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


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


def _weighted_cov(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    mu = x @ w.reshape(-1, 1)
    xc = x - mu
    return xc @ np.diag(w) @ xc.T


def _cov_newey_west(x: np.ndarray, lag: int, half_life: int) -> np.ndarray:
    t = x.shape[1]
    w = _exp_weights(t, half_life)
    cov = _weighted_cov(x, w)

    for k in range(1, lag + 1):
        t2 = t - k
        if t2 <= 1:
            break
        w2 = _exp_weights(t2, half_life)
        x_t = x[:, :t2]
        x_l = x[:, k:]
        mu_t = x_t @ w2.reshape(-1, 1)
        mu_l = x_l @ w2.reshape(-1, 1)
        xc_t = x_t - mu_t
        xc_l = x_l - mu_l
        gamma = xc_t @ np.diag(w2) @ xc_l.T
        bartlett = 1.0 - k / (lag + 1.0)
        cov = cov + bartlett * (gamma + gamma.T)

    return 21.0 * cov


def _eigen_adjust(
    base_cov: np.ndarray,
    lag: int,
    cov_days: int,
    mc: int,
    alpha: float,
    random_seed: Optional[int],
) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(base_cov)
    eigvals = np.clip(eigvals, a_min=1e-10, a_max=None)

    rng = np.random.default_rng(random_seed)
    ratios = []

    for _ in range(max(mc, 1)):
        z = rng.normal(loc=0.0, scale=np.sqrt(eigvals).reshape(-1, 1), size=(len(eigvals), cov_days))
        sim_f = eigvecs @ z
        sim_cov = _cov_newey_west(sim_f, lag=lag, half_life=90)

        s_vals, s_vecs = np.linalg.eigh(sim_cov)
        s_vals = np.clip(s_vals, a_min=1e-10, a_max=None)
        d_tilde = np.diag(s_vecs.T @ base_cov @ s_vecs)
        ratio = np.divide(d_tilde, s_vals, out=np.ones_like(s_vals), where=s_vals > 0)
        ratios.append(ratio)

    lambda_k = np.sqrt(np.mean(ratios, axis=0))
    gamma_k = alpha * (lambda_k - 1.0) + 1.0
    adj_diag = np.diag(np.square(gamma_k))
    return eigvecs @ adj_diag @ eigvecs.T


def _vra_multiplier(x: np.ndarray, base_cov: np.ndarray, short_half_life: int = 21) -> float:
    t = x.shape[1]
    w_short = _exp_weights(t, short_half_life)
    mu_short = x @ w_short.reshape(-1, 1)
    var_short = np.sum(w_short * (x - mu_short) ** 2, axis=1)
    var_long = np.clip(np.diag(base_cov), a_min=1e-12, a_max=None)
    ratio = np.divide(var_short, var_long, out=np.ones_like(var_short), where=var_long > 0)
    mult = float(np.median(ratio))
    return float(np.clip(mult, 0.5, 2.5))


def compute_factor_covariance_matrix(
    factor_returns: pd.DataFrame,
    factor_order: list[str],
    target_date: pd.Timestamp | str,
    lag: int = 5,
    cov_days: int = 252,
    mc: int = 300,
    alpha: float = 1.5,
    nw_half_life: int = 90,
    random_seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    计算单日 Barra 风格因子协方差矩阵（NW + 特征值调整 + VRA）。
    输入 `factor_returns` 为 index=trade_date, columns=因子列。
    """
    if factor_returns.empty:
        raise ValueError("factor_returns 为空")

    missing = [c for c in factor_order if c not in factor_returns.columns]
    if missing:
        raise ValueError(f"factor_returns 缺少因子列: {missing}")

    fr = factor_returns.copy()
    fr.index = pd.to_datetime(fr.index)
    fr = fr.sort_index()

    tgt = pd.to_datetime(target_date)
    if tgt not in fr.index:
        raise ValueError(f"target_date 不在 factor_returns 索引中: {tgt}")

    pos = fr.index.get_loc(tgt)
    if isinstance(pos, slice):
        pos = pos.stop - 1

    start_pos = int(pos) - cov_days + 1
    if start_pos < 0:
        raise ValueError("历史窗口不足，无法计算协方差")

    window = fr.iloc[start_pos : int(pos) + 1][factor_order].T
    x = window.to_numpy(dtype=float)

    base_cov = _cov_newey_west(x, lag=lag, half_life=nw_half_life)
    eig_cov = _eigen_adjust(base_cov, lag=lag, cov_days=cov_days, mc=mc, alpha=alpha, random_seed=random_seed)
    vra = _vra_multiplier(x, base_cov)

    final_cov = eig_cov * vra
    final_cov = (final_cov + final_cov.T) / 2.0
    final_cov = final_cov + np.eye(final_cov.shape[0]) * 1e-8

    return pd.DataFrame(final_cov, index=factor_order, columns=factor_order)
