from __future__ import annotations

import statistics
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def calc_ic(df: pd.DataFrame, factor_col: str, return_col: str = "icfactor") -> Tuple[float, List[float], float]:
    """
    功能: 计算因子逐日 IC 序列、IC 均值与 ICIR。
    Input: df(含因子与收益列的数据), factor_col(因子列), return_col(收益列)。
    Output: (ic_mean, ic_values, icir)。
    """
    grouped = df.groupby("trade_date")
    ic_values: List[float] = []
    for _, sub_df in grouped:
        ic_raw = sub_df[factor_col].corr(sub_df[return_col], method="spearman")
        ic_val = float(ic_raw) if pd.notna(ic_raw) else np.nan
        ic_values.append(ic_val)
    ic_mean = float(np.nanmean(ic_values))
    ic_std = float(np.nanstd(ic_values))
    icir = float(ic_mean / ic_std) if ic_std != 0 else np.nan
    return ic_mean, ic_values, icir


def calc_cumulative_ic(ic_values: List[float], trade_dates: List[pd.Timestamp]) -> pd.Series:
    """
    功能: 计算 IC 累积曲线。
    Input: ic_values(IC 序列), trade_dates(日期序列)。
    Output: 以日期为索引的累计 IC 序列。
    """
    series = pd.Series(data=ic_values, index=pd.to_datetime(trade_dates))
    return series.cumsum()


def plot_ic_curve(ic_values: List[float], trade_dates: List[pd.Timestamp], title: str = "Cumulative IC") -> None:
    """
    功能: 绘制累计 IC 曲线。
    Input: ic_values(IC 序列), trade_dates(日期序列), title(图标题)。
    Output: 无返回；在当前图形后端展示图表。
    """
    cumulative_ic = calc_cumulative_ic(ic_values, trade_dates)
    plt.figure(figsize=(14, 7))
    plt.plot(cumulative_ic, label="Cumulative IC")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative IC")
    plt.legend()
    plt.grid(True)
    plt.show()


def calc_group_nav(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str = "pctchange",
    n_groups: int = 5,
    reverse: bool = False,
) -> Dict[str, pd.Series]:
    """
    功能: 按因子分组计算各组及多空组合净值曲线。
    Input: df(含因子与收益), factor_col(分组因子列), return_col(日收益列), n_groups(分组数), reverse(是否反向多空)。
    Output: 各组与 LongShort 的净值序列字典。
    """
    grouped_daily = [g.copy() for _, g in df.groupby("trade_date")]

    buckets = {f"G{i+1}": [] for i in range(n_groups)}
    long_short = []
    valid_dates = []

    for sub in grouped_daily:
        sub = sub.dropna(subset=[factor_col, return_col])
        if sub.empty:
            continue
        try:
            sub["factor_group"] = pd.qcut(sub[factor_col], q=n_groups, labels=False, duplicates="drop")
        except ValueError:
            continue

        counts = sub["factor_group"].value_counts().sort_index()
        if len(counts) != n_groups:
            continue

        group_returns = []
        for gid in range(n_groups):
            g = sub[sub["factor_group"] == gid]
            r = float(g[return_col].mean() / 100.0 + 1.0)
            group_returns.append(r)
            buckets[f"G{gid+1}"].append(r)

        ls = (group_returns[0] - group_returns[-1] + 1.0) if reverse else (group_returns[-1] - group_returns[0] + 1.0)
        long_short.append(ls)
        valid_dates.append(pd.to_datetime(sub["trade_date"].iloc[0]))

    nav = {}
    for key, vals in buckets.items():
        nav[key] = pd.Series(vals, index=valid_dates).cumprod()
    nav["LongShort"] = pd.Series(long_short, index=valid_dates).cumprod()
    return nav


def calc_perf_stats(nav: pd.Series) -> Dict[str, float]:
    """
    功能: 计算净值序列的年化收益、夏普与最大回撤。
    Input: nav(净值时间序列)。
    Output: 指标字典 {annual_return, sharpe, max_drawdown}。
    """
    if nav.empty or len(nav) < 3:
        return {"annual_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}

    rets = nav.pct_change().dropna()
    if rets.empty:
        return {"annual_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}

    sharpe = np.nan
    var = statistics.variance(rets) if len(rets) >= 2 else 0.0
    if var > 0:
        sharpe = statistics.mean(rets) / np.sqrt(var)

    drawdown = (nav.cummax() - nav) / nav.cummax()
    max_drawdown = float(drawdown.max())

    days = (nav.index[-1] - nav.index[0]).days if len(nav.index) > 1 else 0
    annual_return = np.nan
    if days > 0 and nav.iloc[0] != 0:
        annual_return = float((nav.iloc[-1] / nav.iloc[0]) ** (365.0 / days) - 1.0)

    return {"annual_return": annual_return, "sharpe": float(sharpe) if sharpe is not np.nan else np.nan, "max_drawdown": max_drawdown}


def evaluate_factor(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str = "pctchange",
    ic_return_col: str = "icfactor",
    reverse: bool = False,
) -> Dict[str, object]:
    """
    功能: 对单因子执行完整评估（IC + 分组净值 + 绩效指标）。
    Input: df(评估数据), factor_col(因子列), return_col(收益列), ic_return_col(IC收益列), reverse(是否反向多空)。
    Output: 评估结果字典（含 ic/nav/perf）。
    """
    ic_mean, ic_values, icir = calc_ic(df, factor_col=factor_col, return_col=ic_return_col)
    nav_map = calc_group_nav(df, factor_col=factor_col, return_col=return_col, reverse=reverse)
    stats_map = {k: calc_perf_stats(v) for k, v in nav_map.items()}
    return {
        "ic_mean": ic_mean,
        "ic_values": ic_values,
        "icir": icir,
        "nav": nav_map,
        "perf": stats_map,
    }


def plot_group_nav(nav_map: Dict[str, pd.Series], title: str = "Factor Group NAV") -> None:
    """
    功能: 绘制分组净值与多空净值曲线。
    Input: nav_map(净值字典), title(图标题)。
    Output: 无返回；在当前图形后端展示图表。
    """
    plt.figure(figsize=(14, 7))
    for key, series in nav_map.items():
        plt.plot(np.asarray(series.index), np.asarray(series.values), label=key)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True)
    plt.show()
