from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.risk_covariance import compute_factor_covariance_matrix
from src.risk_specific import align_specific_to_universe, compute_specific_variance_matrix


STYLE_FACTORS = [
    "Size",
    "MIDCAP",
    "BTOP",
    "Liquidity",
    "Leverage",
    "Earnings_Yield",
    "Residual_Volatility",
    "Beta_Factor",
    "Momentum",
    "Growth",
]


def build_daily_exposure_matrix(
    exposure_panel: pd.DataFrame,
    target_date: pd.Timestamp | str,
    include_industry: bool = True,
    industry_col: str = "industry_name",
) -> pd.DataFrame:
    """
    从暴露面板中提取指定日期的暴露矩阵 X（index=stock_code，columns=factor）。
    """
    required = {"stock_code", "trade_date"}
    if not required.issubset(exposure_panel.columns):
        raise ValueError(f"exposure_panel 缺少必要字段: {sorted(required - set(exposure_panel.columns))}")

    panel = exposure_panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    tgt = pd.to_datetime(target_date)

    day = panel[panel["trade_date"] == tgt].copy()
    if day.empty:
        raise ValueError(f"exposure_panel 不包含目标日期数据: {tgt}")

    day = day.drop_duplicates(["stock_code"], keep="last")
    day = day.set_index("stock_code")

    style_cols = [c for c in STYLE_FACTORS if c in day.columns]
    if not style_cols:
        raise ValueError("目标日期没有可用风格因子列")

    x = day[style_cols].astype(float)

    if include_industry and industry_col in day.columns:
        ind = pd.get_dummies(day[industry_col].astype(str), prefix="Ind", dtype=float)
        x = pd.concat([x, ind], axis=1)

    x = x.fillna(0.0)
    x = x.sort_index()
    return x


def validate_risk_inputs(
    factor_returns: pd.DataFrame,
    specific_returns_wide: pd.DataFrame,
    exposure_matrix: pd.DataFrame,
    target_date: pd.Timestamp | str,
) -> None:
    tgt = pd.to_datetime(target_date)

    if factor_returns.empty:
        raise ValueError("factor_returns 为空")
    if specific_returns_wide.empty:
        raise ValueError("specific_returns_wide 为空")
    if exposure_matrix.empty:
        raise ValueError("exposure_matrix 为空")

    fr = factor_returns.copy()
    fr.index = pd.to_datetime(fr.index)
    if tgt not in fr.index:
        raise ValueError(f"factor_returns 缺少目标日期: {tgt}")

    sr = specific_returns_wide.copy()
    sr.columns = pd.to_datetime(sr.columns)
    if tgt not in sr.columns:
        raise ValueError(f"specific_returns_wide 缺少目标日期: {tgt}")


def build_risk_matrices_for_date(
    factor_returns: pd.DataFrame,
    specific_returns_wide: pd.DataFrame,
    exposure_panel: pd.DataFrame,
    target_date: pd.Timestamp | str,
    factor_order: Optional[list[str]] = None,
    include_industry: bool = True,
    cov_days: int = 252,
    nw_lag: int = 5,
    mc: int = 300,
    alpha: float = 1.5,
) -> dict[str, pd.DataFrame]:
    """
    单日构建 X/F/Delta/V。
    返回 dict: {"X", "F", "Delta", "V", "target_date"}
    """
    tgt = pd.to_datetime(target_date)

    x = build_daily_exposure_matrix(
        exposure_panel=exposure_panel,
        target_date=tgt,
        include_industry=include_industry,
    )

    validate_risk_inputs(
        factor_returns=factor_returns,
        specific_returns_wide=specific_returns_wide,
        exposure_matrix=x,
        target_date=tgt,
    )

    if factor_order is None:
        factor_order = x.columns.tolist()

    available = [c for c in factor_order if c in factor_returns.columns and c in x.columns]
    if not available:
        raise ValueError("factor_returns 与 X 无重叠因子列，无法计算 F")

    x_use = x[available].copy()

    f = compute_factor_covariance_matrix(
        factor_returns=factor_returns,
        factor_order=available,
        target_date=tgt,
        lag=nw_lag,
        cov_days=cov_days,
        mc=mc,
        alpha=alpha,
    )

    s_aligned = align_specific_to_universe(
        specific_returns_wide=specific_returns_wide,
        stock_universe=x_use.index,
        fill_value=0.0,
    )

    delta = compute_specific_variance_matrix(
        specific_returns_wide=s_aligned,
        target_date=tgt,
        h=cov_days,
        tau=90,
        lag=nw_lag,
    )

    x_val = x_use.values
    f_val = f.values
    d_val = delta.values

    v = x_val @ f_val @ x_val.T + d_val
    v = (v + v.T) / 2.0
    v = v + np.eye(v.shape[0]) * 1e-8
    v_df = pd.DataFrame(v, index=x_use.index, columns=x_use.index)

    return {
        "X": x_use,
        "F": f,
        "Delta": delta,
        "V": v_df,
        "target_date": pd.DataFrame({"target_date": [tgt]}),
    }


def save_risk_snapshot(
    risk_pack: dict[str, pd.DataFrame],
    snapshot_dir: str | Path,
    target_date: pd.Timestamp | str,
) -> dict[str, str]:
    """
    将单日风险快照落盘为 pkl（X/F/Delta/V），文件名按交易日命名。
    """
    out_dir = Path(snapshot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dt = pd.to_datetime(target_date)
    ds = dt.date().isoformat()

    file_map = {
        "X": out_dir / f"{ds}_X.pkl",
        "F": out_dir / f"{ds}_F.pkl",
        "Delta": out_dir / f"{ds}_Delta.pkl",
        "V": out_dir / f"{ds}_V.pkl",
    }

    for key, fp in file_map.items():
        if key not in risk_pack:
            raise ValueError(f"risk_pack 缺少键: {key}")
        risk_pack[key].to_pickle(fp)

    return {k: str(v) for k, v in file_map.items()}


def build_risk_snapshots_for_rebalance_dates(
    factor_returns: pd.DataFrame,
    specific_returns_wide: pd.DataFrame,
    exposure_panel: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp | str],
    snapshot_dir: str | Path = "./output/risk_snapshots",
    factor_order: Optional[list[str]] = None,
    include_industry: bool = True,
    cov_days: int = 252,
    nw_lag: int = 5,
    mc: int = 300,
    alpha: float = 1.5,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """
    按调仓日批量构建风险快照并落盘为 pkl。
    返回摘要 DataFrame：trade_date / status / message / X/F/Delta/V 文件路径。
    """
    if not rebalance_dates:
        raise ValueError("rebalance_dates 不能为空")

    rows: list[dict[str, object]] = []
    for d in rebalance_dates:
        dt = pd.to_datetime(d)
        try:
            risk_pack = build_risk_matrices_for_date(
                factor_returns=factor_returns,
                specific_returns_wide=specific_returns_wide,
                exposure_panel=exposure_panel,
                target_date=dt,
                factor_order=factor_order,
                include_industry=include_industry,
                cov_days=cov_days,
                nw_lag=nw_lag,
                mc=mc,
                alpha=alpha,
            )
            files = save_risk_snapshot(risk_pack=risk_pack, snapshot_dir=snapshot_dir, target_date=dt)
            rows.append(
                {
                    "trade_date": dt,
                    "status": "ok",
                    "message": "",
                    "X_path": files["X"],
                    "F_path": files["F"],
                    "Delta_path": files["Delta"],
                    "V_path": files["V"],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "trade_date": dt,
                    "status": "error",
                    "message": str(exc),
                    "X_path": None,
                    "F_path": None,
                    "Delta_path": None,
                    "V_path": None,
                }
            )
            if not continue_on_error:
                raise

    summary = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    return summary
