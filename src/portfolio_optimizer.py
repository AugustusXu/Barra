from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class OptimizeConfig:
    risk_aversion: float = 0.25
    single_stock_cap: float = 0.05
    active_weight_cap: float = 0.02
    industry_dev_limit: float = 0.01
    size_restriction_max_active_return: float = 0.05
    max_turnover: float = 0.60
    transaction_cost: float = 0.0015


@dataclass
class ExtendedOptimizeConfig(OptimizeConfig):
    style_neutral_tolerance: float = 0.02
    industry_neutral_tolerance: float = 0.01
    buy_turnover_cap: Optional[float] = None
    sell_turnover_cap: Optional[float] = None
    enforce_long_only: bool = True


def build_trading_constraints(
    weights: cp.Variable,
    prev_weights: Optional[np.ndarray],
    cfg: ExtendedOptimizeConfig,
) -> tuple[list[cp.Constraint], cp.Expression, cp.Expression]:
    """构建交易约束模板，返回(约束, 换手表达式, 成本表达式)。"""
    constraints: list[cp.Constraint] = []
    n = weights.shape[0]

    if cfg.enforce_long_only:
        constraints.append(weights >= 0)

    constraints += [cp.sum(weights) == 1, weights <= cfg.single_stock_cap]

    turnover_expr = cp.Constant(0.0)
    trade_cost_expr = cp.Constant(0.0)

    if prev_weights is not None:
        delta = weights - prev_weights
        buy_expr = cp.sum(cp.pos(delta))
        sell_expr = cp.sum(cp.pos(-delta))
        turnover_expr = (buy_expr + sell_expr) / 2.0
        trade_cost_expr = turnover_expr * cfg.transaction_cost

        if cfg.max_turnover < 1.0:
            constraints.append(turnover_expr <= cfg.max_turnover)
        if cfg.buy_turnover_cap is not None:
            constraints.append(buy_expr / 2.0 <= cfg.buy_turnover_cap)
        if cfg.sell_turnover_cap is not None:
            constraints.append(sell_expr / 2.0 <= cfg.sell_turnover_cap)

    return constraints, turnover_expr, trade_cost_expr


def build_style_neutrality_constraints(
    active_weights: cp.Expression,
    exposure_matrix: pd.DataFrame,
    style_columns: Optional[list[str]] = None,
    industry_columns: Optional[list[str]] = None,
    style_tolerance: float = 0.02,
    industry_tolerance: float = 0.01,
) -> list[cp.Constraint]:
    """风格/行业中性约束模板，约束主动暴露在容忍区间内。"""
    constraints: list[cp.Constraint] = []
    cols = list(exposure_matrix.columns)

    if style_columns is None:
        style_columns = [c for c in cols if not c.startswith("Ind_")]
    if industry_columns is None:
        industry_columns = [c for c in cols if c.startswith("Ind_")]

    if style_columns:
        xs = exposure_matrix[style_columns].to_numpy(dtype=float)
        s_act = xs.T @ active_weights
        constraints += [s_act <= style_tolerance, s_act >= -style_tolerance]

    if industry_columns:
        xi = exposure_matrix[industry_columns].to_numpy(dtype=float)
        i_act = xi.T @ active_weights
        constraints += [i_act <= industry_tolerance, i_act >= -industry_tolerance]

    return constraints


def optimize_portfolio_with_constraints(
    strategy_type: str,
    exposure_matrix: pd.DataFrame,
    total_risk_matrix: pd.DataFrame,
    benchmark_weights: Optional[pd.Series] = None,
    alpha_signal: Optional[pd.Series] = None,
    prev_weights: Optional[pd.Series] = None,
    config: Optional[ExtendedOptimizeConfig] = None,
    enforce_style_neutral_template: bool = True,
    style_columns: Optional[list[str]] = None,
    industry_columns: Optional[list[str]] = None,
) -> dict[str, object]:
    """
    扩展优化入口：新增交易约束与风格中性模板。
    不替代 `optimize_portfolio`，作为增强版本独立使用。
    """
    cfg = config or ExtendedOptimizeConfig()
    if exposure_matrix.empty:
        raise ValueError("exposure_matrix 为空")

    idx = exposure_matrix.index
    n = len(idx)
    x = exposure_matrix.reindex(idx).fillna(0.0)

    v = total_risk_matrix.reindex(index=idx, columns=idx).fillna(0.0).to_numpy(dtype=float)
    v = _ensure_psd(v)

    if benchmark_weights is None:
        w_bench = pd.Series(1.0 / n, index=idx)
    else:
        w_bench = benchmark_weights.reindex(idx).fillna(0.0)
        s = float(w_bench.sum())
        w_bench = (w_bench / s) if s > 0 else pd.Series(1.0 / n, index=idx)

    alpha_vec = _prepare_alpha(alpha_signal, idx)
    w_prev = prev_weights.reindex(idx).fillna(0.0).to_numpy(dtype=float) if prev_weights is not None else None

    w = cp.Variable(n)
    w_active = w - w_bench.to_numpy(dtype=float)

    constraints, turnover_expr, trade_cost_expr = build_trading_constraints(w, w_prev, cfg)
    constraints += [w_active <= cfg.active_weight_cap, w_active >= -cfg.active_weight_cap]

    if enforce_style_neutral_template:
        constraints += build_style_neutrality_constraints(
            active_weights=w_active,
            exposure_matrix=x,
            style_columns=style_columns,
            industry_columns=industry_columns,
            style_tolerance=cfg.style_neutral_tolerance,
            industry_tolerance=cfg.industry_neutral_tolerance,
        )

    risk_abs = cp.quad_form(w, cp.psd_wrap(v))
    risk_active = cp.quad_form(w_active, cp.psd_wrap(v))
    ret_abs = alpha_vec @ w
    ret_active = alpha_vec @ w_active

    if strategy_type == "min_abs_risk":
        objective = cp.Minimize(risk_abs + trade_cost_expr)
    elif strategy_type == "min_active_risk":
        constraints.append(cp.sum(w_active) == 0)
        objective = cp.Minimize(risk_active + trade_cost_expr)
    elif strategy_type == "max_abs_return":
        objective = cp.Maximize(ret_abs - cfg.risk_aversion * risk_abs - trade_cost_expr)
    elif strategy_type == "max_active_return":
        constraints.append(cp.sum(w_active) == 0)
        objective = cp.Maximize(ret_active - cfg.risk_aversion * risk_active - trade_cost_expr)
    else:
        raise ValueError(f"不支持的 strategy_type: {strategy_type}")

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.SCS, verbose=False)
    except Exception:
        problem.solve(solver=cp.ECOS, verbose=False)

    if problem.status != "optimal":
        raise RuntimeError(f"优化失败，status={problem.status}")

    w_opt = pd.Series(np.asarray(w.value).reshape(-1), index=idx).clip(lower=0.0)
    if w_opt.sum() > 0:
        w_opt = w_opt / w_opt.sum()
    w_act = w_opt - w_bench

    return {
        "weights": w_opt,
        "active_weights": w_act,
        "benchmark_weights": w_bench,
        "status": problem.status,
        "objective": float(problem.value) if problem.value is not None else np.nan,
        "turnover": float(turnover_expr.value) if hasattr(turnover_expr, "value") and turnover_expr.value is not None else np.nan,
    }


def _ensure_psd(matrix: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    m = (matrix + matrix.T) / 2.0
    m = m + np.eye(m.shape[0]) * jitter
    return m


def _prepare_alpha(alpha_signal: Optional[pd.Series], index: pd.Index) -> np.ndarray:
    if alpha_signal is None:
        return np.zeros(len(index), dtype=float)
    aligned = alpha_signal.reindex(index).fillna(0.0)
    return aligned.to_numpy(dtype=float)


def optimize_portfolio(
    strategy_type: str,
    exposure_matrix: pd.DataFrame,
    total_risk_matrix: pd.DataFrame,
    benchmark_weights: Optional[pd.Series] = None,
    alpha_signal: Optional[pd.Series] = None,
    prev_weights: Optional[pd.Series] = None,
    config: Optional[OptimizeConfig] = None,
) -> dict[str, object]:
    """
    统一优化入口：
    - min_abs_risk
    - min_active_risk
    - max_abs_return
    - max_active_return
    """
    cfg = config or OptimizeConfig()

    if exposure_matrix.empty:
        raise ValueError("exposure_matrix 为空")

    n = exposure_matrix.shape[0]
    idx = exposure_matrix.index

    v = total_risk_matrix.reindex(index=idx, columns=idx).fillna(0.0).to_numpy(dtype=float)
    v = _ensure_psd(v)

    if benchmark_weights is None:
        w_bench = pd.Series(1.0 / n, index=idx)
    else:
        w_bench = benchmark_weights.reindex(idx).fillna(0.0)
        s = float(w_bench.sum())
        if s <= 0:
            w_bench = pd.Series(1.0 / n, index=idx)
        else:
            w_bench = w_bench / s

    alpha_vec = _prepare_alpha(alpha_signal, idx)

    w_prev = None
    if prev_weights is not None:
        w_prev = prev_weights.reindex(idx).fillna(0.0).to_numpy(dtype=float)

    w_abs = cp.Variable(n)
    w_act = cp.Variable(n)

    obj = None
    constraints: list[cp.Constraint] = []

    if strategy_type == "min_abs_risk":
        risk_expr = cp.quad_form(w_abs, cp.psd_wrap(v))
        obj = risk_expr
        constraints += [w_abs >= 0, cp.sum(w_abs) == 1, w_abs <= cfg.single_stock_cap]

        if w_prev is not None:
            turnover = cp.sum(cp.abs(w_abs - w_prev)) / 2.0
            obj = obj + turnover * cfg.transaction_cost
            if cfg.max_turnover < 1.0:
                constraints.append(turnover <= cfg.max_turnover)

        problem = cp.Problem(cp.Minimize(obj), constraints)

    elif strategy_type == "min_active_risk":
        risk_expr = cp.quad_form(w_act, cp.psd_wrap(v))
        obj = risk_expr
        constraints += [
            cp.sum(w_act) == 0,
            w_act <= cfg.active_weight_cap,
            w_act >= -cfg.active_weight_cap,
            w_act + w_bench.to_numpy(dtype=float) >= 0,
            w_act + w_bench.to_numpy(dtype=float) <= cfg.single_stock_cap,
        ]

        if w_prev is not None:
            turnover = cp.sum(cp.abs((w_act + w_bench.to_numpy(dtype=float)) - w_prev)) / 2.0
            obj = obj + turnover * cfg.transaction_cost
            if cfg.max_turnover < 1.0:
                constraints.append(turnover <= cfg.max_turnover)

        problem = cp.Problem(cp.Minimize(obj), constraints)

    elif strategy_type == "max_abs_return":
        risk_expr = cp.quad_form(w_abs, cp.psd_wrap(v))
        ret_expr = alpha_vec @ w_abs
        obj = ret_expr - cfg.risk_aversion * risk_expr
        constraints += [w_abs >= 0, cp.sum(w_abs) == 1, w_abs <= cfg.single_stock_cap]

        if w_prev is not None:
            turnover = cp.sum(cp.abs(w_abs - w_prev)) / 2.0
            obj = obj - turnover * cfg.transaction_cost
            if cfg.max_turnover < 1.0:
                constraints.append(turnover <= cfg.max_turnover)

        problem = cp.Problem(cp.Maximize(obj), constraints)

    elif strategy_type == "max_active_return":
        risk_expr = cp.quad_form(w_act, cp.psd_wrap(v))
        ret_expr = alpha_vec @ w_act
        obj = ret_expr - cfg.risk_aversion * risk_expr

        constraints += [
            cp.sum(w_act) == 0,
            w_act <= cfg.active_weight_cap,
            w_act >= -cfg.active_weight_cap,
            w_act + w_bench.to_numpy(dtype=float) >= 0,
            w_act + w_bench.to_numpy(dtype=float) <= cfg.single_stock_cap,
        ]

        cols = list(exposure_matrix.columns)
        ind_cols = [c for c in cols if c.startswith("Ind_")]
        if "Size" in cols:
            size_idx = cols.index("Size")
            constraints += [
                exposure_matrix.to_numpy(dtype=float).T @ w_act <= np.where(np.arange(len(cols)) == size_idx, cfg.size_restriction_max_active_return, np.inf),
                exposure_matrix.to_numpy(dtype=float).T @ w_act >= np.where(np.arange(len(cols)) == size_idx, -cfg.size_restriction_max_active_return, -np.inf),
            ]

        if ind_cols:
            exp_act = exposure_matrix[ind_cols].to_numpy(dtype=float).T @ w_act
            constraints += [exp_act <= cfg.industry_dev_limit, exp_act >= -cfg.industry_dev_limit]

        if w_prev is not None:
            turnover = cp.sum(cp.abs((w_act + w_bench.to_numpy(dtype=float)) - w_prev)) / 2.0
            obj = obj - turnover * cfg.transaction_cost
            if cfg.max_turnover < 1.0:
                constraints.append(turnover <= cfg.max_turnover)

        problem = cp.Problem(cp.Maximize(obj), constraints)

    else:
        raise ValueError(f"不支持的 strategy_type: {strategy_type}")

    try:
        problem.solve(solver=cp.SCS, verbose=False)
    except Exception:
        problem.solve(solver=cp.ECOS, verbose=False)

    if problem.status != "optimal":
        raise RuntimeError(f"优化失败，status={problem.status}")

    if "active" in strategy_type:
        w_opt = pd.Series(np.asarray(w_act.value).reshape(-1) + w_bench.to_numpy(dtype=float), index=idx)
        w_active = pd.Series(np.asarray(w_act.value).reshape(-1), index=idx)
    else:
        w_opt = pd.Series(np.asarray(w_abs.value).reshape(-1), index=idx)
        w_active = w_opt - w_bench

    w_opt = w_opt.clip(lower=0.0)
    if w_opt.sum() > 0:
        w_opt = w_opt / w_opt.sum()

    return {
        "weights": w_opt,
        "active_weights": w_active,
        "benchmark_weights": w_bench,
        "status": problem.status,
        "objective": float(problem.value) if problem.value is not None else np.nan,
    }
