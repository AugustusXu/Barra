from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.data_loader import load_pctchange_daily_pkl

import numpy as np
import pandas as pd
import statsmodels.api as sm


KEY_COLS = {"stock_code", "trade_date", "next_return", "pctchange", "icfactor", "ts_code"}


def _to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _infer_factor_col(df: pd.DataFrame, factor_name: str) -> Optional[str]:
    if factor_name in df.columns:
        return factor_name
    numeric_cols: List[str] = [
        col
        for col in df.columns
        if col not in KEY_COLS and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_cols:
        return None
    return numeric_cols[0]


def add_next_return(
    pctchange_df: pd.DataFrame,
    stock_col: str = "stock_code",
    date_col: str = "trade_date",
    return_col: str = "pctchange",
    label_col: str = "next_return",
) -> pd.DataFrame:
    """
    功能: 将当期收益序列转换为下一期收益标签。
    Input: pctchange_df(收益表), stock/date/return/label 列名。
    Output: 新增 next_return 标签列后的 DataFrame。
    """
    data = pctchange_df.copy()
    data[date_col] = _to_datetime_safe(data[date_col])
    data = data.dropna(subset=[date_col])
    data = data.sort_values([stock_col, date_col])
    data[label_col] = data.groupby(stock_col)[return_col].shift(-1)
    return data


def build_exposure_panel(
    factor_data: Dict[str, pd.DataFrame],
    selected_factors: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    功能: 将多个因子结果表合并为统一暴露面板。
    Input: factor_data(因子名->DataFrame), selected_factors(可选因子名单)。
    Output: (暴露面板, 实际使用的因子列名列表)。
    """
    if selected_factors is None:
        factor_names = list(factor_data.keys())
    else:
        factor_names = [name for name in selected_factors if name in factor_data]

    panel: Optional[pd.DataFrame] = None
    factor_cols_used: List[str] = []

    for factor_name in factor_names:
        df = factor_data.get(factor_name)
        if df is None or df.empty:
            continue
        if not {"stock_code", "trade_date"}.issubset(df.columns):
            continue

        tmp = df.copy()
        tmp["trade_date"] = _to_datetime_safe(tmp["trade_date"])
        tmp = tmp.dropna(subset=["trade_date"])

        factor_col = _infer_factor_col(tmp, factor_name)
        if factor_col is None:
            continue

        out_col = factor_name
        tmp = tmp[["stock_code", "trade_date", factor_col]].rename(columns={factor_col: out_col})
        tmp = tmp.drop_duplicates(["stock_code", "trade_date"], keep="last")

        if panel is None:
            panel = tmp
        else:
            panel = panel.merge(tmp, on=["stock_code", "trade_date"], how="outer")
        factor_cols_used.append(out_col)

    if panel is None:
        panel = pd.DataFrame(columns=["stock_code", "trade_date"])

    panel = panel.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
    return panel, factor_cols_used


def compute_daily_factor_returns(
    panel_df: pd.DataFrame,
    factor_cols: List[str],
    weight_col: Optional[str] = None,
    min_stocks: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    功能: 按日做截面回归，计算因子收益与特异性收益。
    Input: panel_df(含 next_return 和因子暴露), factor_cols(因子列列表), weight_col(可选权重列), min_stocks(最小样本数)。
    Output: (因子日收益 DataFrame, 特异性收益长表 DataFrame)。
    """
    if "next_return" not in panel_df.columns:
        raise ValueError("panel_df 缺少 next_return 列")

    work = panel_df.copy()
    work["trade_date"] = _to_datetime_safe(work["trade_date"])
    work = work.dropna(subset=["trade_date"])

    factor_ret_rows: List[pd.Series] = []
    specific_rows: List[pd.DataFrame] = []

    for trade_date, sub in work.groupby("trade_date"):
        needed = factor_cols + ["next_return"]
        if weight_col is not None and weight_col in sub.columns:
            needed.append(weight_col)
        sub = sub.dropna(subset=needed)

        if len(sub) < min_stocks:
            continue

        X = sub[factor_cols]
        y = sub["next_return"]

        try:
            if weight_col is not None and weight_col in sub.columns:
                w = pd.to_numeric(sub[weight_col], errors="coerce")
                w = w.clip(lower=0.0)
                w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                w = np.sqrt(w)
                model = sm.WLS(y, X, weights=w)
            else:
                model = sm.OLS(y, X)
            result = model.fit()
        except Exception:
            continue

        params = result.params.reindex(factor_cols)
        params.name = trade_date
        factor_ret_rows.append(params)

        resid_df = pd.DataFrame(
            {
                "trade_date": trade_date,
                "stock_code": sub["stock_code"].values,
                "specific_return": result.resid.values,
            }
        )
        specific_rows.append(resid_df)

    factor_returns = pd.DataFrame(factor_ret_rows)
    if not factor_returns.empty:
        factor_returns.index.name = "trade_date"
        factor_returns = factor_returns.sort_index()

    specific_returns = pd.concat(specific_rows, ignore_index=True) if specific_rows else pd.DataFrame(
        columns=["trade_date", "stock_code", "specific_return"]
    )
    return factor_returns, specific_returns


def run_factor_return_pipeline(
    factor_data: Dict[str, pd.DataFrame],
    pctchange_daily_dir: str | Path,
    output_root: str | Path = "./output",
    selected_factors: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    buffer_days: int = 0,
    market_cap_df: Optional[pd.DataFrame] = None,
    market_cap_col: str = "s_dq_mv",
    min_stocks: int = 30,
) -> Dict[str, object]:
    """
    功能: 端到端计算因子收益率并落盘到 output。
    Input: 因子数据字典、按日收益目录、输出目录、筛选参数、可选市值权重。
    Output: 包含路径与结果 DataFrame 的字典。
    """
    exposure_panel, factor_cols = build_exposure_panel(factor_data, selected_factors=selected_factors)
    if exposure_panel.empty or not factor_cols:
        raise ValueError("无法构建有效的因子暴露面板，请检查 factor_data")

    pct_df = load_pctchange_daily_pkl(
        pctchange_daily_dir=pctchange_daily_dir,
        start_date=start_date,
        end_date=end_date,
        buffer_days=buffer_days,
    )
    pct_df = add_next_return(pct_df, label_col="next_return")

    panel = exposure_panel.merge(
        pct_df[["stock_code", "trade_date", "next_return"]],
        on=["stock_code", "trade_date"],
        how="inner",
    )

    if market_cap_df is not None and not market_cap_df.empty:
        cap = market_cap_df.copy()
        rename_map = {}
        if "ts_code" in cap.columns and "stock_code" not in cap.columns:
            rename_map["ts_code"] = "stock_code"
        if rename_map:
            cap = cap.rename(columns=rename_map)
        if {"stock_code", "trade_date", market_cap_col}.issubset(cap.columns):
            cap = cap[["stock_code", "trade_date", market_cap_col]].copy()
            cap["trade_date"] = _to_datetime_safe(cap["trade_date"])
            cap = cap.dropna(subset=["trade_date"])
            cap = cap.drop_duplicates(["stock_code", "trade_date"], keep="last")
            panel = panel.merge(cap, on=["stock_code", "trade_date"], how="left")
            weight_col = market_cap_col
        else:
            weight_col = None
    else:
        weight_col = None

    factor_returns, specific_returns = compute_daily_factor_returns(
        panel_df=panel,
        factor_cols=factor_cols,
        weight_col=weight_col,
        min_stocks=min_stocks,
    )

    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    factor_csv = out_root / "factor_daily_returns.csv"
    factor_pkl = out_root / "factor_daily_returns.pkl"
    specific_csv = out_root / "specific_returns_long.csv"
    specific_pkl = out_root / "specific_returns_long.pkl"

    #factor_returns.to_csv(factor_csv)
    factor_returns.to_pickle(factor_pkl)
    #specific_returns.to_csv(specific_csv, index=False)
    specific_returns.to_pickle(specific_pkl)

    return {
        "factor_returns": factor_returns,
        "specific_returns": specific_returns,
        "factor_returns_csv": str(factor_csv),
        "factor_returns_pkl": str(factor_pkl),
        "specific_returns_csv": str(specific_csv),
        "specific_returns_pkl": str(specific_pkl),
        "factor_cols": factor_cols,
    }
