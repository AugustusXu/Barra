from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RiskAttributionResult:
    total_variance: float
    factor_variance: float
    specific_variance: float
    style_variance: float
    industry_variance: float
    style_ratio: float
    industry_ratio: float
    specific_ratio: float


def _safe_scalar(x: np.ndarray | float) -> float:
    val = float(np.asarray(x).reshape(-1)[0]) if np.asarray(x).size > 0 else float(x)
    return 0.0 if np.isnan(val) else val


def attribute_portfolio_risk(
    weights: pd.Series,
    exposure_matrix: pd.DataFrame,
    factor_cov: pd.DataFrame,
    specific_var_diag: pd.DataFrame,
    benchmark_weights: Optional[pd.Series] = None,
    style_prefixes: tuple[str, ...] = ("Size", "Beta", "Momentum", "ResidualVolatility", "Liquidity", "NonLinearSize", "BookToPrice", "EarningsYield", "Growth", "Leverage"),
    industry_prefix: str = "Ind_",
) -> dict[str, object]:
    """
    风险归因（方差口径）：
    - 绝对风险：w
    - 主动风险：w - w_bench
    分解：风格/行业/特异。
    """
    if weights.empty:
        raise ValueError("weights 为空")

    idx = weights.index
    x = exposure_matrix.reindex(index=idx).fillna(0.0)
    f = factor_cov.reindex(index=x.columns, columns=x.columns).fillna(0.0)

    d = specific_var_diag.reindex(index=idx, columns=idx).fillna(0.0)

    w = weights.reindex(idx).fillna(0.0).to_numpy(dtype=float)

    if benchmark_weights is None:
        wb = np.zeros_like(w)
    else:
        wb = benchmark_weights.reindex(idx).fillna(0.0).to_numpy(dtype=float)

    wa = w - wb

    x_np = x.to_numpy(dtype=float)
    f_np = ((f.to_numpy(dtype=float) + f.to_numpy(dtype=float).T) / 2.0)
    d_np = np.diag(np.diag(d.to_numpy(dtype=float)))

    total_var = _safe_scalar(wa.T @ (x_np @ f_np @ x_np.T + d_np) @ wa)
    factor_var = _safe_scalar(wa.T @ (x_np @ f_np @ x_np.T) @ wa)
    specific_var = _safe_scalar(wa.T @ d_np @ wa)

    cols = list(x.columns)
    style_cols = [c for c in cols if any(c.startswith(p) for p in style_prefixes)]
    industry_cols = [c for c in cols if c.startswith(industry_prefix)]

    style_var = 0.0
    if style_cols:
        xs = x[style_cols].to_numpy(dtype=float)
        fs = f.reindex(index=style_cols, columns=style_cols).fillna(0.0).to_numpy(dtype=float)
        fs = (fs + fs.T) / 2.0
        style_var = _safe_scalar(wa.T @ (xs @ fs @ xs.T) @ wa)

    industry_var = 0.0
    if industry_cols:
        xi = x[industry_cols].to_numpy(dtype=float)
        fi = f.reindex(index=industry_cols, columns=industry_cols).fillna(0.0).to_numpy(dtype=float)
        fi = (fi + fi.T) / 2.0
        industry_var = _safe_scalar(wa.T @ (xi @ fi @ xi.T) @ wa)

    denom = total_var if abs(total_var) > 1e-16 else np.nan

    result = RiskAttributionResult(
        total_variance=total_var,
        factor_variance=factor_var,
        specific_variance=specific_var,
        style_variance=style_var,
        industry_variance=industry_var,
        style_ratio=float(style_var / denom) if denom == denom else np.nan,
        industry_ratio=float(industry_var / denom) if denom == denom else np.nan,
        specific_ratio=float(specific_var / denom) if denom == denom else np.nan,
    )

    return {
        "summary": pd.Series(result.__dict__),
        "active_weights": pd.Series(wa, index=idx),
    }


def factor_risk_contributions(
    weights: pd.Series,
    exposure_matrix: pd.DataFrame,
    factor_cov: pd.DataFrame,
    benchmark_weights: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    输出按因子粒度的主动风险贡献明细（方差口径）。
    返回列：`active_exposure`, `marginal`, `contribution`, `contribution_ratio`。
    """
    if weights.empty:
        raise ValueError("weights 为空")

    idx = weights.index
    x = exposure_matrix.reindex(index=idx).fillna(0.0)
    f = factor_cov.reindex(index=x.columns, columns=x.columns).fillna(0.0)

    w = weights.reindex(idx).fillna(0.0).to_numpy(dtype=float)
    if benchmark_weights is None:
        wb = np.zeros_like(w)
    else:
        wb = benchmark_weights.reindex(idx).fillna(0.0).to_numpy(dtype=float)

    wa = w - wb
    x_np = x.to_numpy(dtype=float)
    f_np = (f.to_numpy(dtype=float) + f.to_numpy(dtype=float).T) / 2.0

    b = x_np.T @ wa
    marginal = f_np @ b
    contribution = b * marginal
    total = float(np.sum(contribution))

    out = pd.DataFrame(
        {
            "active_exposure": b,
            "marginal": marginal,
            "contribution": contribution,
        },
        index=x.columns,
    )
    out.index.name = "factor"
    out["contribution_ratio"] = out["contribution"] / total if abs(total) > 1e-16 else np.nan
    return out.sort_values("contribution", ascending=False)


def attribute_risk_over_time(
    weights_by_date: pd.DataFrame,
    exposure_by_date: dict[pd.Timestamp, pd.DataFrame],
    factor_cov_by_date: dict[pd.Timestamp, pd.DataFrame],
    specific_var_by_date: dict[pd.Timestamp, pd.DataFrame],
    benchmark_by_date: Optional[pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    """
    时序风险归因。
    - `weights_by_date`: index=trade_date, columns=stock_code
    - *_by_date: key=trade_date
    返回：
    - `summary_ts`: 每期汇总归因
    - `factor_detail_ts`: 因子粒度长表明细
    """
    if weights_by_date.empty:
        raise ValueError("weights_by_date 为空")

    summary_rows: list[pd.Series] = []
    detail_rows: list[pd.DataFrame] = []

    for dt in pd.to_datetime(weights_by_date.index):
        d = pd.Timestamp(dt)
        if d not in exposure_by_date or d not in factor_cov_by_date or d not in specific_var_by_date:
            continue

        w = weights_by_date.loc[d]
        x = exposure_by_date[d]
        f = factor_cov_by_date[d]
        delta = specific_var_by_date[d]

        wb = None
        if benchmark_by_date is not None and d in pd.to_datetime(benchmark_by_date.index):
            wb = benchmark_by_date.loc[d]

        summary = attribute_portfolio_risk(
            weights=w,
            exposure_matrix=x,
            factor_cov=f,
            specific_var_diag=delta,
            benchmark_weights=wb,
        )["summary"]
        summary.name = d
        summary_rows.append(summary)

        fac_detail = factor_risk_contributions(
            weights=w,
            exposure_matrix=x,
            factor_cov=f,
            benchmark_weights=wb,
        ).reset_index()
        fac_detail["trade_date"] = d
        detail_rows.append(fac_detail)

    summary_ts = pd.DataFrame(summary_rows)
    if not summary_ts.empty:
        summary_ts.index.name = "trade_date"

    factor_detail_ts = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    if not factor_detail_ts.empty:
        factor_detail_ts = factor_detail_ts[["trade_date", "factor", "active_exposure", "marginal", "contribution", "contribution_ratio"]]

    return {
        "summary_ts": summary_ts,
        "factor_detail_ts": factor_detail_ts,
    }
