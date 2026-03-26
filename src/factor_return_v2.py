from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data_loader import load_pctchange_daily_pkl
import numpy as np
import pandas as pd


KEY_COLS = {"stock_code", "trade_date", "next_return", "pctchange", "icfactor", "ts_code"}


def _to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _infer_factor_col(df: pd.DataFrame, factor_name: str) -> Optional[str]:
    if factor_name in df.columns:
        return factor_name
    numeric_cols = [
        col for col in df.columns if col not in KEY_COLS and pd.api.types.is_numeric_dtype(df[col])
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


def _prepare_industry(industry_df: pd.DataFrame) -> pd.DataFrame:
    if industry_df.empty:
        return pd.DataFrame(columns=["stock_code", "trade_date", "industry_name"])

    data = industry_df.copy()
    rename_map = {}
    if "ts_code" in data.columns and "stock_code" not in data.columns:
        rename_map["ts_code"] = "stock_code"
    if rename_map:
        data = data.rename(columns=rename_map)

    industry_col = "L1name" if "L1name" in data.columns else "industry_name"
    if industry_col not in data.columns:
        return pd.DataFrame(columns=["stock_code", "trade_date", "industry_name"])

    req = ["stock_code", "trade_date", industry_col]
    if not set(req).issubset(data.columns):
        return pd.DataFrame(columns=["stock_code", "trade_date", "industry_name"])

    data = data[req].copy()
    data = data.rename(columns={industry_col: "industry_name"})
    data["trade_date"] = _to_datetime_safe(data["trade_date"])
    data = data.dropna(subset=["trade_date"])
    data["industry_name"] = data["industry_name"].astype(str).fillna("未知行业")
    data = data.drop_duplicates(["stock_code", "trade_date"], keep="last")
    return data


def _winsorize_and_standardize(series: pd.Series, mad_k: float = 5.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    median = values.median(skipna=True)
    abs_dev = (values - median).abs()
    mad = abs_dev.median(skipna=True)
    if pd.isna(mad) or mad == 0:
        mad = 1e-6
    lower = median - mad_k * 1.4826 * mad
    upper = median + mad_k * 1.4826 * mad
    clipped = values.clip(lower=lower, upper=upper)

    mean = clipped.mean(skipna=True)
    std = clipped.std(skipna=True)
    if pd.isna(std) or std == 0:
        std = 1.0
    return (clipped - mean) / std


def _impute_style_exposure(sub_df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    data = sub_df.copy()
    for col in factor_cols:
        col_numeric = pd.to_numeric(data[col], errors="coerce") if col in data.columns else pd.Series(np.nan, index=data.index)

        if "industry_name" in data.columns:
            ind_median = col_numeric.groupby(data["industry_name"]).transform("median")
            col_numeric = col_numeric.fillna(ind_median)

        col_numeric = col_numeric.fillna(col_numeric.median(skipna=True))
        col_numeric = col_numeric.fillna(0.0)

        data[col] = _winsorize_and_standardize(col_numeric)
    return data


def _prepare_weights(sub_df: pd.DataFrame, weight_col: Optional[str], scheme: str = "cap") -> Optional[pd.Series]:
    if weight_col is None or weight_col not in sub_df.columns:
        return None

    w = pd.to_numeric(sub_df[weight_col], errors="coerce")
    w = w.replace([np.inf, -np.inf], np.nan)
    w = w.fillna(w.median(skipna=True))
    w = w.fillna(0.0)
    w = w.clip(lower=0.0)

    if scheme == "sqrt_cap":
        w = w.pow(0.5)
    elif scheme == "equal":
        w = pd.Series(1.0, index=sub_df.index)

    low = w.quantile(0.01)
    high = w.quantile(0.99)
    if pd.notna(low) and pd.notna(high) and high > low:
        w = w.clip(lower=low, upper=high)

    mean_val = float(w.mean())
    if pd.isna(mean_val) or mean_val <= 0:
        return None
    out = w / mean_val
    return pd.Series(out, index=sub_df.index)


def compute_daily_factor_returns_v2(
    panel_df: pd.DataFrame,
    factor_cols: List[str],
    industry_col: str = "industry_name",
    weight_col: Optional[str] = None,
    min_stocks: int = 50,
    weight_scheme: str = "cap",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    修复点:
    2) 含行业项（行业哑变量）
    3) 对风格因子做按日去极值+标准化
    4) 加截距项（与drop_first行业哑变量共同保证可识别）
    5) 明确WLS权重口径（默认cap，可选sqrt_cap/equal）
    6) 降低样本选择偏差（仅对收益/行业硬过滤，因子缺失做分层插补）
    """
    if "next_return" not in panel_df.columns:
        raise ValueError("panel_df 缺少 next_return 列")

    work = panel_df.copy()
    work["trade_date"] = _to_datetime_safe(work["trade_date"])
    work = work.dropna(subset=["trade_date", "next_return", industry_col])

    factor_ret_rows: List[pd.Series] = []
    specific_rows: List[pd.DataFrame] = []

    def _fit_cs_regression(x_df: pd.DataFrame, y_ser: pd.Series, w_ser: Optional[pd.Series]) -> Tuple[pd.Series, pd.Series]:
        x_mat = x_df.to_numpy(dtype=float)
        y_vec = y_ser.to_numpy(dtype=float)

        if w_ser is not None:
            w_vec = np.asarray(w_ser.to_numpy(dtype=float), dtype=float)
            w_vec = np.clip(w_vec, a_min=0.0, a_max=None)
            sqrt_w = np.sqrt(w_vec)
            x_w = x_mat * sqrt_w[:, None]
            y_w = y_vec * sqrt_w
            beta = np.linalg.pinv(x_w.T @ x_w) @ (x_w.T @ y_w)
        else:
            beta = np.linalg.pinv(x_mat.T @ x_mat) @ (x_mat.T @ y_vec)

        pred = x_mat @ beta
        resid = y_vec - pred

        params = pd.Series(beta, index=x_df.columns)
        resid_series = pd.Series(resid, index=x_df.index)
        return params, resid_series

    for trade_date, sub in work.groupby("trade_date"):
        sub = sub.copy()
        if len(sub) < min_stocks:
            continue

        for col in factor_cols:
            if col not in sub.columns:
                sub[col] = np.nan

        sub = _impute_style_exposure(sub, factor_cols)
        industry_dummy = pd.get_dummies(sub[industry_col].astype(str), prefix="IND", drop_first=True, dtype=float)

        style_exposure = sub[factor_cols].copy()
        X = pd.concat([style_exposure, industry_dummy], axis=1)
        X = pd.DataFrame(X, index=sub.index)
        X.insert(0, "const", 1.0)
        y = pd.Series(pd.to_numeric(sub["next_return"], errors="coerce"), index=sub.index, name="next_return")

        reg_df = pd.concat([sub[["stock_code"]], X, y.to_frame()], axis=1).dropna()
        if len(reg_df) < min_stocks:
            continue

        X_reg = reg_df.loc[:, X.columns]
        y_reg = pd.Series(reg_df["next_return"], index=reg_df.index)

        weights = _prepare_weights(reg_df, weight_col=weight_col, scheme=weight_scheme)

        try:
            params, resid = _fit_cs_regression(X_reg, y_reg, weights)
        except Exception:
            continue

        style_ret = params.reindex(factor_cols)
        const_val = params.get("const", np.nan)
        style_ret.loc["Intercept"] = const_val
        style_ret.name = trade_date
        factor_ret_rows.append(style_ret)

        specific_rows.append(
            pd.DataFrame(
                {
                    "trade_date": trade_date,
                    "stock_code": reg_df["stock_code"].values,
                    "specific_return": resid.values,
                }
            )
        )

    factor_returns = pd.DataFrame(factor_ret_rows)
    if not factor_returns.empty:
        factor_returns.index.name = "trade_date"
        factor_returns = factor_returns.sort_index()

    specific_returns = pd.concat(specific_rows, ignore_index=True) if specific_rows else pd.DataFrame(
        columns=["trade_date", "stock_code", "specific_return"]
    )

    return factor_returns, specific_returns


def run_factor_return_pipeline_v2(
    factor_data: Dict[str, pd.DataFrame],
    industry_df: pd.DataFrame,
    pctchange_daily_dir: str | Path,
    output_root: str | Path = "./output",
    selected_factors: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    buffer_days: int = 0,
    market_cap_df: Optional[pd.DataFrame] = None,
    market_cap_col: str = "s_dq_mv",
    min_stocks: int = 50,
    weight_scheme: str = "cap",
) -> Dict[str, object]:
    exposure_panel, factor_cols = build_exposure_panel(factor_data, selected_factors=selected_factors)
    if exposure_panel.empty or not factor_cols:
        raise ValueError("无法构建有效的因子暴露面板，请检查 factor_data")

    ind = _prepare_industry(industry_df)
    panel = exposure_panel.merge(ind, on=["stock_code", "trade_date"], how="left")
    panel["industry_name"] = panel["industry_name"].fillna("未知行业")

    pct_df = load_pctchange_daily_pkl(
        pctchange_daily_dir=pctchange_daily_dir,
        start_date=start_date,
        end_date=end_date,
        buffer_days=buffer_days,
    )
    pct_df = add_next_return(pct_df, label_col="next_return")

    panel = panel.merge(
        pct_df[["stock_code", "trade_date", "next_return"]],
        on=["stock_code", "trade_date"],
        how="inner",
    )

    weight_col = None
    if market_cap_df is not None and not market_cap_df.empty:
        cap = market_cap_df.copy()
        if "ts_code" in cap.columns and "stock_code" not in cap.columns:
            cap = cap.rename(columns={"ts_code": "stock_code"})

        if {"stock_code", "trade_date", market_cap_col}.issubset(cap.columns):
            cap = cap[["stock_code", "trade_date", market_cap_col]].copy()
            cap["trade_date"] = _to_datetime_safe(cap["trade_date"])
            cap = cap.dropna(subset=["trade_date"])
            cap = cap.drop_duplicates(["stock_code", "trade_date"], keep="last")
            panel = panel.merge(cap, on=["stock_code", "trade_date"], how="left")
            weight_col = market_cap_col

    factor_returns, specific_returns = compute_daily_factor_returns_v2(
        panel_df=panel,
        factor_cols=factor_cols,
        industry_col="industry_name",
        weight_col=weight_col,
        min_stocks=min_stocks,
        weight_scheme=weight_scheme,
    )

    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    factor_pkl = out_root / "factor_daily_returns_v2.pkl"
    specific_pkl = out_root / "specific_returns_long_v2.pkl"

    factor_returns.to_pickle(factor_pkl)
    specific_returns.to_pickle(specific_pkl)

    return {
        "factor_returns": factor_returns,
        "specific_returns": specific_returns,
        "factor_returns_pkl": str(factor_pkl),
        "specific_returns_pkl": str(specific_pkl),
        "factor_cols": factor_cols,
    }
