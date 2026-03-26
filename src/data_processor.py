from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import zscore


def mad_winsorize_series(x: pd.Series, multiplier: float = 5.0) -> pd.Series:
    """
    功能: 对序列执行 MAD 去极值并做标准化。
    Input: x(因子序列), multiplier(MAD 截断倍数)。
    Output: 处理后的标准化序列。
    """
    series = pd.to_numeric(x, errors="coerce")
    arr = series.to_numpy(dtype=float)
    median = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - median)))
    if mad == 0 or np.isnan(mad):
        mad = 1e-6
    upper = float(median + multiplier * 1.4826 * mad)
    lower = float(median - multiplier * 1.4826 * mad)
    clipped = series.clip(lower=lower, upper=upper)
    std = float(np.nanstd(clipped.to_numpy(dtype=float)))
    if std == 0 or np.isnan(std):
        return pd.Series(clipped.to_numpy(dtype=float) - float(np.nanmean(clipped.to_numpy(dtype=float))), index=series.index)
    return pd.Series((clipped.to_numpy(dtype=float) - float(np.nanmean(clipped.to_numpy(dtype=float)))) / std, index=series.index)


def standardize_by_date(df: pd.DataFrame, factor_col: str, date_col: str = "trade_date") -> pd.DataFrame:
    """
    功能: 按日期截面对指定因子列做标准化。
    Input: df(原始表), factor_col(因子列), date_col(日期列)。
    Output: 标准化后的 DataFrame。
    """
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    result = result.dropna(subset=[date_col])
    result[factor_col] = result.groupby(date_col)[factor_col].transform(mad_winsorize_series)
    return result


def remove_outliers_and_zscore(df: pd.DataFrame, factor_col: str, date_col: str = "trade_date") -> pd.DataFrame:
    """
    功能: 按日期分组去极值并执行 zscore 标准化。
    Input: df(原始表), factor_col(因子列), date_col(日期列)。
    Output: 去极值+标准化后的 DataFrame。
    """
    data = df.copy()

    def _clip_and_z(group: pd.DataFrame) -> pd.DataFrame:
        """
        功能: 对单日截面因子进行截断与 zscore。
        Input: group(单个日期的截面数据)。
        Output: 处理后的单日 DataFrame。
        """
        vals = pd.to_numeric(group[factor_col], errors="coerce").to_numpy(dtype=float)
        median = float(np.nanmedian(vals))
        mad = float(np.nanmedian(np.abs(vals - median)))
        lower = float(median - 5 * mad)
        upper = float(median + 5 * mad)
        group[factor_col] = pd.to_numeric(group[factor_col], errors="coerce").clip(lower=lower, upper=upper)
        group[factor_col] = zscore(group[factor_col], nan_policy="omit")
        return group

    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])
    data = data.groupby(data[date_col]).apply(_clip_and_z).reset_index(drop=True)
    return data


def process_updown_st_year1(data: pd.DataFrame) -> pd.DataFrame:
    """
    功能: 过滤一字板、ST 以及上市未满一年股票。
    Input: data(含行情与上市信息的数据表)。
    Output: 过滤后的 DataFrame。
    """
    df = data.copy()
    if {"open", "close"}.issubset(df.columns):
        df = df[~(df["open"] == df["close"])]
    if "black_list_tag" in df.columns:
        df = df[~(df["black_list_tag"] == "yes")]
    if {"trade_date", "S_INFO_LISTDATE"}.issubset(df.columns):
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df["S_INFO_LISTDATE"] = pd.to_datetime(df["S_INFO_LISTDATE"], errors="coerce")
        listed_days = df["trade_date"] - df["S_INFO_LISTDATE"]
        df = df[listed_days >= pd.Timedelta(days=365)]
    return df


def fill_quarterly_to_daily(
    daily_df: pd.DataFrame,
    low_freq_df: pd.DataFrame,
    factor_col: str,
    stock_col: str = "stock_code",
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """
    功能: 将低频财务因子按股票向前填充到日频表。
    Input: daily_df(日频基表), low_freq_df(低频因子表), factor_col(待填充因子列), stock_col/date_col(键列)。
    Output: 合并并填充后的日频 DataFrame。
    """
    left = daily_df.copy()
    right = low_freq_df.copy()
    left[date_col] = pd.to_datetime(left[date_col], errors="coerce")
    right[date_col] = pd.to_datetime(right[date_col], errors="coerce")

    merged = pd.merge(left, right[[stock_col, date_col, factor_col]], how="left", on=[stock_col, date_col])
    merged = merged.sort_values([stock_col, date_col])
    merged[factor_col] = merged.groupby(stock_col)[factor_col].ffill()
    return merged


def add_next_return_label(
    price_df: pd.DataFrame,
    stock_col: str = "stock_code",
    return_col: str = "pctchange",
    label_col: str = "icfactor",
) -> pd.DataFrame:
    """
    功能: 构造下一期收益标签列（如 IC 使用的 icfactor）。
    Input: price_df(价格/收益表), stock_col(股票列), return_col(收益列), label_col(新标签列名)。
    Output: 新增标签列后的 DataFrame。
    """
    data = price_df.copy()
    data = data.sort_values([stock_col, "trade_date"])
    data[label_col] = data.groupby(stock_col)[return_col].shift(-1)
    return data


def align_tables_on_keys(tables: List[pd.DataFrame], keys: Iterable[str] = ("stock_code", "trade_date")) -> pd.DataFrame:
    """
    功能: 按主键将多张表外连接对齐。
    Input: tables(DataFrame 列表), keys(对齐键列)。
    Output: 合并后的 DataFrame。
    """
    if not tables:
        raise ValueError("tables 不能为空")

    result = tables[0].copy()
    key_cols = list(keys)
    for frame in tables[1:]:
        result = result.merge(frame, on=key_cols, how="outer")
    return result


def align_specific_returns_to_exposure(
    specific_returns_wide: pd.DataFrame,
    exposure_index: pd.Index,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    功能: 将特异性收益宽表按暴露矩阵股票顺序对齐。
    Input: specific_returns_wide(宽表), exposure_index(目标股票索引), fill_value(缺失填充值)。
    Output: 对齐后的特异性收益宽表。
    """
    aligned = specific_returns_wide.reindex(exposure_index)
    return aligned.fillna(fill_value)


def preprocess_exposure_cross_section(
    exposure_df: pd.DataFrame,
    style_cols: List[str],
    industry_col: str = "industry_name",
) -> pd.DataFrame:
    """
    功能: 对单日风格暴露做行业中位数插补与截面去极值标准化。
    Input: exposure_df(单日暴露表), style_cols(风格列), industry_col(行业列)。
    Output: 处理后的单日暴露 DataFrame。
    """
    data = exposure_df.copy()

    for col in style_cols:
        if col not in data.columns:
            data[col] = np.nan

        values = pd.to_numeric(data[col], errors="coerce")
        if industry_col in data.columns:
            ind_med = values.groupby(data[industry_col]).transform("median")
            values = values.fillna(ind_med)

        values = values.fillna(values.median(skipna=True))
        values = values.fillna(0.0)
        data[col] = mad_winsorize_series(values)

    return data
