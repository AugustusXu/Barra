from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data_processor import mad_winsorize_series


# -----------------------------
# Common Helpers
# -----------------------------
def _ensure_datetime(df: pd.DataFrame, date_col: str = "trade_date") -> pd.DataFrame:
    """
    功能: 将日期列转换为 datetime 并剔除空日期行。
    Input: df(原始表), date_col(日期列名)。
    Output: 日期标准化后的 DataFrame。
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out.dropna(subset=[date_col])


def _ewma_weights(window: int, half_life: int, normalize: bool = True) -> np.ndarray:
    """
    功能: 生成指数衰减权重序列。
    Input: window(窗口长度), half_life(半衰期), normalize(是否归一化)。
    Output: 一维 numpy 权重数组。
    """
    w = 0.5 ** (np.arange(window)[::-1] / half_life)
    return w / w.sum() if normalize else w


def _pivot(price_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    功能: 将长表转换为 trade_date × stock_code 宽表。
    Input: price_df(长表), value_col(值列名)。
    Output: 透视后的 DataFrame。
    """
    return pd.pivot_table(price_df, values=value_col, index="trade_date", columns="stock_code")


def _safe_inverse(series: pd.Series) -> pd.Series:
    """
    功能: 对序列做安全倒数（0 转 NaN）。
    Input: series(数值序列)。
    Output: 倒数序列。
    """
    x = pd.to_numeric(series, errors="coerce")
    x = x.replace(0, np.nan)
    return 1.0 / x


def _cross_sectional_z(df: pd.DataFrame, col: str, date_col: str = "trade_date") -> pd.Series:
    """
    功能: 按日期截面对因子列进行标准化。
    Input: df(含因子表), col(因子列), date_col(日期列)。
    Output: 标准化后的 Series。
    """
    return df.groupby(date_col)[col].transform(mad_winsorize_series)


# -----------------------------
# Size
# -----------------------------
def compute_size_factors(ashareeodderivativeindicator: pd.DataFrame) -> pd.DataFrame:
    """
    功能: 计算 Size 模块的 LNCAP、MIDCAP 与合成 Size 因子。
    Input: ashareeodderivativeindicator(含 s_dq_mv 的行情衍生表)。
    Output: 列为 stock_code/trade_date/LNCAP/MIDCAP/Size 的 DataFrame。
    """
    data = ashareeodderivativeindicator[["stock_code", "trade_date", "s_dq_mv"]].copy()
    data = _ensure_datetime(data)
    data = data.dropna(subset=["s_dq_mv"])

    data["s_dq_mv"] = data["s_dq_mv"] / 10000.0
    data["LNCAP_raw"] = np.log(data["s_dq_mv"] + 1.0)
    data["LNCAP"] = _cross_sectional_z(data, "LNCAP_raw")
    data["LNCAP_cubed"] = data["LNCAP"] ** 3

    def _orth(group: pd.DataFrame) -> pd.Series:
        """
        功能: 在单日截面对 LNCAP^3 对 LNCAP 做回归并取残差标准化。
        Input: group(单日截面数据)。
        Output: 与 group 索引对齐的 MIDCAP 残差序列。
        """
        y = group["LNCAP_cubed"].values
        x = sm.add_constant(group["LNCAP"].values)
        w = np.sqrt(group["s_dq_mv"].values)
        try:
            resid = sm.WLS(y, x, weights=w).fit().resid
        except Exception:
            resid = np.zeros_like(y)
        return pd.Series(mad_winsorize_series(pd.Series(resid, index=group.index)), index=group.index)

    data["MIDCAP"] = data.groupby("trade_date", group_keys=False).apply(_orth)
    data["Size"] = 0.5 * data["LNCAP"] + 0.5 * data["MIDCAP"]
    return data[["stock_code", "trade_date", "LNCAP", "MIDCAP", "Size"]]


# -----------------------------
# Volatility
# -----------------------------
def compute_volatility_factors(
    stock_price: pd.DataFrame,
    index_price: pd.DataFrame,
    market_code: str = "000300.SH",
    window: int = 252,
    half_life: int = 63,
) -> pd.DataFrame:
    """
    功能: 计算波动率相关因子（BETA/Hist_sigma/Daily_std）及合成 Volatility。
    Input: stock_price(个股行情), index_price(指数行情), market_code(市场代码), window/half_life(参数)。
    Output: 含波动率描述词与合成因子的 DataFrame。
    """
    stock = _ensure_datetime(stock_price)
    index_ = _ensure_datetime(index_price)

    stock = stock[["stock_code", "trade_date", "close", "preclose", "pctchange"]].copy()
    index_ = index_[["stock_code", "trade_date", "close", "preclose", "pctchange"]].copy()
    stock["ret"] = stock["pctchange"] / 100.0
    index_["ret"] = index_["pctchange"] / 100.0

    panel = pd.concat([stock, index_], ignore_index=True)
    ret = _pivot(panel, "ret").sort_index()

    if market_code not in ret.columns:
        raise ValueError(f"波动率计算缺少市场指数列: {market_code}")

    w = _ewma_weights(window=window, half_life=half_life)
    w_diag = np.diag(w)

    rows = []
    for i in range(window - 1, len(ret)):
        block = ret.iloc[i - window + 1 : i + 1]
        day = block.index[-1]
        mkt = block[market_code]

        sub = block.drop(columns=[market_code])
        sub = sub.loc[:, sub.isna().mean() <= 0.5]
        if sub.empty:
            continue

        valid = sub.dropna(axis=1)
        if valid.empty:
            continue

        m_vals = mkt.fillna(0.0).values
        y = valid.fillna(0.0).values

        sum_w = np.sum(w)
        sum_wm = np.dot(w, m_vals)
        sum_wmm = np.dot(w, m_vals**2)
        sum_wy = w @ y
        sum_wmy = (w * m_vals) @ y
        sum_wyy = w @ (y**2)

        det = sum_w * sum_wmm - sum_wm**2

        if det == 0:
            alpha_v = sum_wy / sum_w if sum_w != 0 else np.zeros(y.shape[1])
            beta_v = np.zeros(y.shape[1])
        else:
            alpha_v = (sum_wmm * sum_wy - sum_wm * sum_wmy) / det
            beta_v = (sum_w * sum_wmy - sum_wm * sum_wy) / det

        alpha_s = pd.Series(alpha_v, index=valid.columns)
        beta_s = pd.Series(beta_v, index=valid.columns)

        # resid = y - (alpha + beta * m)
        # hsig^2 = sum(w * resid^2) / sum(w)
        hsig2_sq = (sum_wyy
                    + alpha_v**2 * sum_w
                    + beta_v**2 * sum_wmm
                    - 2 * alpha_v * sum_wy
                    - 2 * beta_v * sum_wmy
                    + 2 * alpha_v * beta_v * sum_wm) / sum_w

        hsig_v = np.sqrt(np.maximum(hsig2_sq, 0.0))
        hsig = pd.Series(hsig_v, index=valid.columns)

        day_ret = block.iloc[-1].drop(index=market_code, errors="ignore")
        dstd = block.drop(columns=[market_code]).std(axis=0, ddof=1)

        for code in valid.columns:
            rows.append(
                {
                    "trade_date": day,
                    "stock_code": code,
                    "Hist_alpha": alpha_s.get(code, np.nan),
                    "BETA": beta_s.get(code, np.nan),
                    "Hist_sigma": hsig.get(code, np.nan),
                    "Daily_std": dstd.get(code, np.nan),
                    "_ret_day": day_ret.get(code, np.nan),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Z_BETA"] = _cross_sectional_z(out, "BETA")
    out["Z_Hist_sigma"] = _cross_sectional_z(out, "Hist_sigma")
    out["Z_Daily_std"] = _cross_sectional_z(out, "Daily_std")

    out["Residual_Volatility"] = 0.74 * out["Z_Daily_std"] + 0.10 * out["Z_Hist_sigma"]
    out["Beta_Factor"] = out["Z_BETA"]
    out["Volatility"] = 0.5 * out["Residual_Volatility"] + 0.5 * out["Beta_Factor"]

    return out.drop(columns=["_ret_day"], errors="ignore")


# -----------------------------
# Liquidity
# -----------------------------
def compute_liquidity_factors(ashareeodderivativeindicator: pd.DataFrame) -> pd.DataFrame:
    """
    功能: 计算流动性描述词 STOM/STOQ/STOA 并合成 Liquidity。
    Input: ashareeodderivativeindicator(含 s_dq_turn)。
    Output: 含流动性描述词、标准化值和 Liquidity 的 DataFrame。
    """
    data = ashareeodderivativeindicator[["stock_code", "trade_date", "s_dq_turn"]].copy()
    data = _ensure_datetime(data)
    data["turnover_ratio"] = pd.to_numeric(data["s_dq_turn"], errors="coerce") / 100.0

    pivot = pd.pivot_table(data, index="trade_date", columns="stock_code", values="turnover_ratio").fillna(0)
    stom = pd.DataFrame(np.log(pivot.rolling(21).sum() + 1e-6), index=pivot.index, columns=pivot.columns)
    stoq = pd.DataFrame(np.log(pivot.rolling(63).sum() / 3.0 + 1e-6), index=pivot.index, columns=pivot.columns)
    stoa = pd.DataFrame(np.log(pivot.rolling(252).sum() / 12.0 + 1e-6), index=pivot.index, columns=pivot.columns)

    fac = (
        stom.reset_index().melt(id_vars="trade_date", value_name="STOM")
        .merge(stoq.reset_index().melt(id_vars="trade_date", value_name="STOQ"), on=["trade_date", "stock_code"])
        .merge(stoa.reset_index().melt(id_vars="trade_date", value_name="STOA"), on=["trade_date", "stock_code"])
        .dropna()
    )

    fac["Z_STOM"] = _cross_sectional_z(fac, "STOM")
    fac["Z_STOQ"] = _cross_sectional_z(fac, "STOQ")
    fac["Z_STOA"] = _cross_sectional_z(fac, "STOA")
    fac["Liquidity"] = 0.35 * fac["Z_STOM"] + 0.35 * fac["Z_STOQ"] + 0.30 * fac["Z_STOA"]
    return fac


# -----------------------------
# Momentum A / B / C
# -----------------------------
def compute_momentum_A(price_df: pd.DataFrame, window: int = 504, half_life: int = 126, lag: int = 21) -> pd.DataFrame:
    """
    功能: Momentum A 版本（结构化）计算 RSTR 与 Momentum。
    Input: price_df(收益或价格表), window/half_life/lag(滚动参数)。
    Output: 列为 trade_date/stock_code/RSTR/Z_RSTR/Momentum 的 DataFrame。
    """
    price = _ensure_datetime(price_df)
    if "ret" not in price.columns:
        if {"close", "preclose"}.issubset(price.columns):
            price["ret"] = price["close"] / price["preclose"] - 1
        elif "pctchange" in price.columns:
            price["ret"] = price["pctchange"] / 100.0
        else:
            raise ValueError("Momentum A 需要 ret 或 close/preclose 或 pctchange")

    ret = _pivot(price, "ret").sort_index()
    log_ret = pd.DataFrame(np.log1p(ret), index=ret.index, columns=ret.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    w = _ewma_weights(window, half_life)
    r = np.full_like(log_ret.values, np.nan, dtype=float)

    for i in range(window + lag - 1, len(log_ret)):
        block = log_ret.values[i - lag - window + 1 : i - lag + 1, :]
        r[i, :] = np.dot(w, block)

    rstr = pd.DataFrame(r, index=log_ret.index, columns=log_ret.columns)
    out = rstr.reset_index().melt(id_vars="trade_date", value_name="RSTR").dropna()
    out["Z_RSTR"] = _cross_sectional_z(out, "RSTR")
    out["Momentum"] = out["Z_RSTR"]
    return out[["trade_date", "stock_code", "RSTR", "Z_RSTR", "Momentum"]]


def compute_momentum_B(
    price_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    cap_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    功能: Momentum B 版本（全量）计算 RSTR/STREV/Seasonality/INDMOM 与 Momentum。
    Input: price_df(价格收益表), industry_df(行业表), cap_df(市值表)。
    Output: 含全量动量描述词与 Momentum 的 DataFrame。
    """
    price = _ensure_datetime(price_df)
    ind = _ensure_datetime(industry_df)
    cap = _ensure_datetime(cap_df)

    if "ret" not in price.columns:
        if {"close", "preclose"}.issubset(price.columns):
            price["ret"] = price["close"] / price["preclose"] - 1
        elif "pctchange" in price.columns:
            price["ret"] = price["pctchange"] / 100.0
        else:
            raise ValueError("Momentum B 需要 ret 或 close/preclose 或 pctchange")

    ret = _pivot(price, "ret").sort_index()
    log_ret = pd.DataFrame(np.log1p(ret), index=ret.index, columns=ret.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # STREV
    w_strev = _ewma_weights(21, 5)
    strev_vals = np.full_like(log_ret.values, np.nan, dtype=float)
    for i in range(20, len(log_ret)):
        strev_vals[i, :] = np.dot(w_strev, log_ret.values[i - 20 : i + 1, :])
    strev = pd.DataFrame(strev_vals, index=log_ret.index, columns=log_ret.columns)

    # RSTR
    w_rstr = _ewma_weights(504, 126)
    rstr_vals = np.full_like(log_ret.values, np.nan, dtype=float)
    for i in range(504 + 21 - 1, len(log_ret)):
        block = log_ret.values[i - 21 - 504 + 1 : i - 21 + 1, :]
        rstr_vals[i, :] = np.dot(w_rstr, block)
    rstr = pd.DataFrame(rstr_vals, index=log_ret.index, columns=log_ret.columns)

    # Seasonality
    ret_21 = log_ret.rolling(21).sum()
    seasonality = (ret_21.shift(252) + ret_21.shift(504) + ret_21.shift(756) + ret_21.shift(1008) + ret_21.shift(1260)) / 5.0

    base = rstr.reset_index().melt(id_vars="trade_date", value_name="RSTR").dropna()
    base = base.merge(
        strev.reset_index().melt(id_vars="trade_date", value_name="STREV"),
        on=["trade_date", "stock_code"],
        how="left",
    )
    base = base.merge(
        seasonality.reset_index().melt(id_vars="trade_date", value_name="Seasonality"),
        on=["trade_date", "stock_code"],
        how="left",
    )

    ind_col = "L1name" if "L1name" in ind.columns else ("industry_name" if "industry_name" in ind.columns else None)
    if ind_col is None:
        raise ValueError("industry_df 需要 L1name 或 industry_name 字段")

    base = base.merge(ind[["stock_code", "trade_date", ind_col]], on=["stock_code", "trade_date"], how="left")
    base = base.merge(cap[["stock_code", "trade_date", "s_dq_mv"]], on=["stock_code", "trade_date"], how="left")
    base = base.dropna(subset=["RSTR", "s_dq_mv", ind_col])

    base["cap_sqrt"] = np.sqrt(base["s_dq_mv"].clip(lower=0))
    base["weighted_RSTR"] = base["RSTR"] * base["cap_sqrt"]

    ind_rs = (
        base.groupby(["trade_date", ind_col])
        .apply(lambda x: x["weighted_RSTR"].sum() / (x["cap_sqrt"].sum() + 1e-6))
        .reset_index(name="Ind_RSTR")
    )
    base = base.merge(ind_rs, on=["trade_date", ind_col], how="left")
    base["INDMOM"] = base["Ind_RSTR"] - base["RSTR"]

    base["Z_RSTR"] = _cross_sectional_z(base, "RSTR")
    base["Momentum"] = base["Z_RSTR"]
    return base[["trade_date", "stock_code", "Momentum", "RSTR", "Z_RSTR", "STREV", "Seasonality", "INDMOM", "Ind_RSTR"]]


def compute_momentum_C(
    price_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    cap_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    功能: Momentum C 版本，同时输出 A 风格和 B 风格结果。
    Input: price_df(价格收益表), industry_df(行业表), cap_df(市值表)。
    Output: (risk_df, full_df) 二元组。
    """
    a = compute_momentum_A(price_df)
    b = compute_momentum_B(price_df, industry_df, cap_df)
    risk = a[["trade_date", "stock_code", "Momentum"]].copy()
    full = b.copy()
    return risk, full


# -----------------------------
# Quality
# -----------------------------
def compute_quality_leverage(
    ashareeodderivativeindicator: pd.DataFrame,
    asharebalancesheet: pd.DataFrame,
    asharefinancialindicator: pd.DataFrame,
) -> pd.DataFrame:
    """
    功能: 计算 Leverage 模块（Market/Book/Debt）并合成 Leverage。
    Input: 三张基础表（衍生指标/资产负债表/财务指标）。
    Output: 含各杠杆描述词及 Leverage 的 DataFrame。
    """
    lev = _ensure_datetime(ashareeodderivativeindicator[["stock_code", "trade_date", "s_val_mv", "s_val_pb_new"]].copy())
    bs = _ensure_datetime(
        asharebalancesheet[["stock_code", "trade_date", "other_equity_tools_p_shr", "tot_assets", "tot_liab", "tot_non_cur_liab"]].copy()
    )
    fin = _ensure_datetime(asharefinancialindicator[["stock_code", "trade_date", "s_fa_debttoassets"]].copy())

    fin = fin.rename(columns={"s_fa_debttoassets": "Debt_to_asset_ratio"})
    fin["Debt_to_asset_ratio"] = pd.to_numeric(fin["Debt_to_asset_ratio"], errors="coerce") / 100.0

    fac = lev.merge(bs, on=["stock_code", "trade_date"], how="left").merge(fin, on=["stock_code", "trade_date"], how="left")
    cols = ["other_equity_tools_p_shr", "tot_assets", "tot_liab", "tot_non_cur_liab", "Debt_to_asset_ratio"]
    fac = fac.sort_values(["stock_code", "trade_date"])
    fac[cols] = fac.groupby("stock_code")[cols].ffill()

    fac["ME"] = fac["s_val_mv"] * 10000.0
    fac["s_val_pb_new"] = fac["s_val_pb_new"].replace(0, np.nan).fillna(1.0)
    fac["BE"] = fac["ME"] / fac["s_val_pb_new"]
    fac["LD"] = fac["tot_non_cur_liab"].fillna(0)
    fac["PE"] = fac["other_equity_tools_p_shr"].fillna(0)

    fac["Market_Leverage"] = 1.0 + (fac["PE"] + fac["LD"]) / fac["ME"].replace(0, np.nan)
    fac["Book_Leverage"] = 1.0 + (fac["PE"] + fac["LD"]) / (fac["BE"] + 1e-6)

    for col in ["Market_Leverage", "Book_Leverage", "Debt_to_asset_ratio"]:
        fac[f"Z_{col}"] = _cross_sectional_z(fac, col)

    fac["Leverage"] = 0.38 * fac["Z_Market_Leverage"] + 0.35 * fac["Z_Debt_to_asset_ratio"] + 0.27 * fac["Z_Book_Leverage"]
    return fac[["trade_date", "stock_code", "Market_Leverage", "Book_Leverage", "Debt_to_asset_ratio", "Leverage"]]


def compute_quality_earnings_variability(
    ashareincome: pd.DataFrame,
    asharecashflow: pd.DataFrame,
    fy0: Optional[pd.DataFrame],
    pctchange: pd.DataFrame,
) -> pd.DataFrame:
    """
    功能: 计算盈利波动质量因子（销售/利润/现金流波动与预期项）。
    Input: ashareincome、asharecashflow、fy0、pctchange。
    Output: 含 Earnings_Variability 的 DataFrame。
    """
    sales = _ensure_datetime(ashareincome[["stock_code", "trade_date", "oper_rev"]].copy())
    prof = _ensure_datetime(asharecashflow[["stock_code", "trade_date", "net_profit", "net_incr_cash_cash_equ"]].copy())

    sales["Variation_in_Sales"] = sales.groupby("stock_code")["oper_rev"].rolling(20, min_periods=1).std().reset_index(level=0, drop=True)
    prof["Variation_in_Earning"] = prof.groupby("stock_code")["net_profit"].rolling(10, min_periods=1).std().reset_index(level=0, drop=True)
    prof["Variation_in_Cashflow"] = prof.groupby("stock_code")["net_incr_cash_cash_equ"].rolling(20, min_periods=1).std().reset_index(level=0, drop=True)

    fac = sales[["stock_code", "trade_date", "Variation_in_Sales"]].merge(
        prof[["stock_code", "trade_date", "Variation_in_Earning", "Variation_in_Cashflow"]],
        on=["stock_code", "trade_date"],
        how="outer",
    )

    fac["Earnings_Variability"] = (
        fac[["Variation_in_Sales", "Variation_in_Earning", "Variation_in_Cashflow"]].mean(axis=1)
    )

    if fy0 is not None and not fy0.empty:
        f = _ensure_datetime(fy0.copy())
        date_col = "est_dt" if "est_dt" in f.columns else "trade_date"
        if date_col != "trade_date":
            f = f.rename(columns={date_col: "trade_date"})
        if "est_eps" in f.columns:
            px = _ensure_datetime(pctchange[["stock_code", "trade_date", "close"]].copy())
            eps = f[["stock_code", "trade_date", "est_eps"]].merge(px, on=["stock_code", "trade_date"], how="left")
            eps["forecast_EP_std"] = eps["est_eps"] / eps["close"].replace(0, np.nan)
            fac = fac.merge(eps[["stock_code", "trade_date", "forecast_EP_std"]], on=["stock_code", "trade_date"], how="left")
            fac["Earnings_Variability"] = fac["Earnings_Variability"].fillna(0) + 0.25 * fac["forecast_EP_std"].fillna(0)

    return fac


def compute_quality_earnings_quality(
    asharebalancesheet: pd.DataFrame,
    asharecashflow: pd.DataFrame,
    asharefinancialindicator: pd.DataFrame,
) -> pd.DataFrame:
    """
    功能: 计算 Earnings Quality（ABS/ACF）并合成 Earnings_Quality。
    Input: 资产负债表、现金流量表、财务指标表。
    Output: 含 ABS/ACF/Earnings_Quality 的 DataFrame。
    """
    a = _ensure_datetime(
        asharebalancesheet[
            [
                "stock_code",
                "trade_date",
                "tot_assets",
                "tot_liab",
                "tot_non_cur_liab",
                "deferred_exp",
                "non_cur_liab_due_within_1y",
            ]
        ].copy()
    )
    b = _ensure_datetime(
        asharecashflow[
            [
                "stock_code",
                "trade_date",
                "cash_cash_equ_end_period",
                "depr_fa_coga_dpba",
                "amort_intang_assets",
                "net_cash_flows_oper_act",
                "net_cash_flows_inv_act",
                "net_profit",
            ]
        ].copy()
    )
    c = _ensure_datetime(asharefinancialindicator[["stock_code", "trade_date", "s_fa_interestdebt"]].copy())

    fac = a.merge(b, on=["stock_code", "trade_date"], how="outer").merge(c, on=["stock_code", "trade_date"], how="outer").fillna(0)
    fac = fac.rename(
        columns={
            "tot_assets": "TA",
            "tot_liab": "TL",
            "tot_non_cur_liab": "NCL",
            "deferred_exp": "DEF",
            "non_cur_liab_due_within_1y": "NCL1Y",
            "cash_cash_equ_end_period": "Cash",
            "depr_fa_coga_dpba": "Dep",
            "amort_intang_assets": "Amort",
            "net_cash_flows_oper_act": "CFO",
            "net_cash_flows_inv_act": "CFI",
            "net_profit": "NI",
            "s_fa_interestdebt": "DebtInt",
        }
    )

    fac = fac.sort_values(["stock_code", "trade_date"])
    fac["TD"] = fac["NCL1Y"] + fac["NCL"] + fac["DebtInt"]
    fac["DA"] = fac["Dep"] + fac["Amort"] + fac["DEF"]
    fac["NOA"] = (fac["TA"] - fac["Cash"]) - (fac["TL"] - fac["TD"])
    fac["delta_NOA"] = fac.groupby("stock_code")["NOA"].diff()
    fac["ACCR_BS"] = fac["delta_NOA"] - fac["DA"]
    fac["ABS"] = -fac["ACCR_BS"] / fac["TA"].replace(0, np.nan)

    fac["ACCR_CF"] = fac["NI"] - (fac["CFO"] + fac["CFI"]) + fac["DA"]
    fac["ACF"] = -fac["ACCR_CF"] / fac["TA"].replace(0, np.nan)
    fac["Earnings_Quality"] = 0.5 * (fac["ABS"] + fac["ACF"])
    return fac[["stock_code", "trade_date", "ABS", "ACF", "Earnings_Quality"]].dropna(subset=["Earnings_Quality"])


def compute_quality_profitability(
    asharefinancialindicator: pd.DataFrame,
    ashareincome: pd.DataFrame,
    asharebalancesheet: pd.DataFrame,
) -> pd.DataFrame:
    """
    功能: 计算盈利能力相关描述词并合成 Profitability。
    Input: asharefinancialindicator、ashareincome、asharebalancesheet。
    Output: 含 Profitability 及底层描述词的 DataFrame。
    """
    fin = _ensure_datetime(
        asharefinancialindicator[["stock_code", "trade_date", "s_fa_assetsturn", "s_fa_roa", "s_fa_grossmargin"]].copy()
    )
    inc = _ensure_datetime(ashareincome[["stock_code", "trade_date", "oper_rev"]].copy())
    bs = _ensure_datetime(asharebalancesheet[["stock_code", "trade_date", "tot_assets"]].copy())

    gp = fin[["stock_code", "trade_date", "s_fa_grossmargin"]].merge(bs, on=["stock_code", "trade_date"], how="left")
    gp["Gross_profitability"] = gp["s_fa_grossmargin"] / gp["tot_assets"].replace(0, np.nan)

    gpm = fin[["stock_code", "trade_date", "s_fa_grossmargin"]].merge(inc, on=["stock_code", "trade_date"], how="left")
    gpm["Gross_profit_margin"] = gpm["s_fa_grossmargin"] / gpm["oper_rev"].replace(0, np.nan)

    fac = fin[["stock_code", "trade_date", "s_fa_assetsturn", "s_fa_roa"]].merge(
        gp[["stock_code", "trade_date", "Gross_profitability"]],
        on=["stock_code", "trade_date"],
        how="left",
    )
    fac = fac.merge(gpm[["stock_code", "trade_date", "Gross_profit_margin"]], on=["stock_code", "trade_date"], how="left")
    fac["Profitability"] = fac[["s_fa_assetsturn", "s_fa_roa", "Gross_profit_margin", "Gross_profitability"]].mean(axis=1)
    return fac


def compute_quality_investment(
    asharebalancesheet: pd.DataFrame,
    ashareeodderivativeindicator: pd.DataFrame,
    asharecashflow: pd.DataFrame,
) -> pd.DataFrame:
    """
    功能: 计算投资质量成长率并合成 InvestmentQuality。
    Input: 资产负债表、行情衍生指标表、现金流表。
    Output: 含三类成长率和 InvestmentQuality 的 DataFrame。
    """
    ta = _ensure_datetime(asharebalancesheet[["stock_code", "trade_date", "tot_assets"]].copy())
    fa = _ensure_datetime(ashareeodderivativeindicator[["stock_code", "trade_date", "float_a_shr_today"]].copy())
    capex = _ensure_datetime(asharecashflow[["stock_code", "trade_date", "cash_pay_acq_const_fiolta"]].copy())

    def _growth_rate(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
        """
        功能: 计算单一指标的分组增长率。
        Input: df(指标表), value_col(原值列), out_col(输出列名)。
        Output: 含增长率列的 DataFrame。
        """
        t = df[["stock_code", "trade_date", value_col]].copy().sort_values(["stock_code", "trade_date"])
        t[out_col] = t.groupby("stock_code")[value_col].pct_change(2)
        return t[["stock_code", "trade_date", out_col]]

    g1 = _growth_rate(ta, "tot_assets", "Total_Assets_Growth_Rate")
    g2 = _growth_rate(fa, "float_a_shr_today", "float_a_shr_Growth_Rate")
    g3 = _growth_rate(capex, "cash_pay_acq_const_fiolta", "cash_pay_acq_const_fiolta_Growth_Rate")

    fac = g1.merge(g2, on=["stock_code", "trade_date"], how="outer").merge(g3, on=["stock_code", "trade_date"], how="outer")
    fac["InvestmentQuality"] = fac[
        ["Total_Assets_Growth_Rate", "float_a_shr_Growth_Rate", "cash_pay_acq_const_fiolta_Growth_Rate"]
    ].mean(axis=1)
    return fac


# -----------------------------
# Value
# -----------------------------
def compute_value_btop(ashareeodderivativeindicator: pd.DataFrame) -> pd.DataFrame:
    """
    功能: 计算 Value 模块 BTOP 因子（BP 标准化）。
    Input: ashareeodderivativeindicator(含 s_val_pb_new)。
    Output: 含 BP/BTOP 的 DataFrame。
    """
    pb = _ensure_datetime(ashareeodderivativeindicator[["stock_code", "trade_date", "s_val_pb_new"]].copy())
    pb["BP"] = _safe_inverse(pb["s_val_pb_new"])
    pb = pb.dropna(subset=["BP"])
    pb["BTOP"] = _cross_sectional_z(pb, "BP")
    return pb[["stock_code", "trade_date", "BP", "BTOP"]]


def compute_value_earnings_yield(
    ashareeodderivativeindicator: pd.DataFrame,
    asharefinancialindicator: pd.DataFrame,
    asharebalancesheet: pd.DataFrame,
    asharecashflow: pd.DataFrame,
    fy0: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    功能: 计算 Earnings Yield 模块（ETOP/CETOP/EM）并合成 Earnings_Yield。
    Input: 衍生指标、财务指标、资产负债表、现金流、FY0 数据。
    Output: 含 Earnings_Yield 及底层描述词的 DataFrame。
    """
    base = _ensure_datetime(
        ashareeodderivativeindicator[["stock_code", "trade_date", "s_val_mv", "s_val_pe", "s_val_pcf_ncf"]].copy()
    )
    ebit = _ensure_datetime(asharefinancialindicator[["stock_code", "trade_date", "s_fa_ebit"]].copy())
    liab = _ensure_datetime(asharebalancesheet[["stock_code", "trade_date", "tot_liab"]].copy())
    cash = _ensure_datetime(asharecashflow[["stock_code", "trade_date", "cash_cash_equ_end_period"]].copy())

    fac = base.merge(ebit, on=["stock_code", "trade_date"], how="left")
    fac = fac.merge(liab, on=["stock_code", "trade_date"], how="left")
    fac = fac.merge(cash, on=["stock_code", "trade_date"], how="left")

    fac = fac.sort_values(["stock_code", "trade_date"])
    fac[["s_fa_ebit", "tot_liab", "cash_cash_equ_end_period"]] = fac.groupby("stock_code")[
        ["s_fa_ebit", "tot_liab", "cash_cash_equ_end_period"]
    ].ffill()

    fac["ETOP"] = _safe_inverse(fac["s_val_pe"])
    fac["PCF"] = _safe_inverse(fac["s_val_pcf_ncf"])

    #fac["CETOP"] = np.nan
    #if fy0 is not None and not fy0.empty:
    #    f = _ensure_datetime(fy0.copy())
    #    if "est_dt" in f.columns:
    #        f = f.rename(columns={"est_dt": "trade_date"})
    #    if "est_pe" in f.columns:
    #        f["CETOP"] = _safe_inverse(f["est_pe"])
    #        fac = fac.merge(f[["stock_code", "trade_date", "CETOP"]], on=["stock_code", "trade_date"], how="left")


    fac["CETOP"] = np.nan
    if fy0 is not None and not fy0.empty:
        f = _ensure_datetime(fy0.copy())
        if "est_dt" in f.columns:
            f = f.rename(columns={"est_dt": "trade_date"})
        if "est_pe" in f.columns:
            f["CETOP"] = _safe_inverse(f["est_pe"])
            fac = fac.merge(
                f[["stock_code", "trade_date", "CETOP"]],
                on=["stock_code", "trade_date"],
                how="left",
                suffixes=("_base", "_fy0"),
            )
            fac["CETOP"] = fac["CETOP_base"].combine_first(fac["CETOP_fy0"])
            fac.drop(columns=["CETOP_base", "CETOP_fy0"], inplace=True)



    fac["ME"] = fac["s_val_mv"] * 10000.0
    denom = fac["ME"] + fac["tot_liab"].fillna(0) - fac["cash_cash_equ_end_period"].fillna(0)
    fac["EM"] = fac["s_fa_ebit"] / denom.replace(0, np.nan)

    for col in ["ETOP", "CETOP", "EM"]:
        fac[f"Z_{col}"] = _cross_sectional_z(fac, col)

    fac["Earnings_Yield"] = 0.68 * fac["Z_CETOP"].fillna(0) + 0.11 * fac["Z_ETOP"].fillna(0) + 0.21 * fac["Z_EM"].fillna(0)
    return fac[["trade_date", "stock_code", "ETOP", "CETOP", "PCF", "EM", "Earnings_Yield"]]


def compute_value_long_term_reversal(
    stock_price: pd.DataFrame,
    index_price: pd.DataFrame,
    market_code: str = "000300.SH",
    window: int = 750,
    half_life: int = 260,
) -> pd.DataFrame:
    """
    功能: 计算长期反转（Longterm_Relative_strength、Longterm_Alpha、Long_term_reversal）。
    Input: stock_price(个股), index_price(指数), market_code/window/half_life(参数)。
    Output: 长期反转因子 DataFrame。
    """
    stock = _ensure_datetime(stock_price)
    index_ = _ensure_datetime(index_price)

    stock = stock[["stock_code", "trade_date", "close", "preclose"]].copy()
    index_ = index_[["stock_code", "trade_date", "close", "preclose"]].copy()

    stock["ret"] = np.log(stock["close"] / stock["preclose"])
    index_["ret"] = np.log(index_["close"] / index_["preclose"])

    rs_ret = _pivot(stock, "ret").sort_index()
    w = _ewma_weights(window, half_life)

    rs_rows = []
    for i in range(window - 1, len(rs_ret)):
        block = rs_ret.iloc[i - window + 1 : i + 1].fillna(0)
        score = np.dot(w, block.values)
        date = block.index[-1]
        for code, val in zip(block.columns, score):
            rs_rows.append({"trade_date": date, "stock_code": code, "Longterm_Relative_strength": -val})

    rs_df = pd.DataFrame(rs_rows)

    panel = pd.concat([stock, index_], ignore_index=True)
    ret = _pivot(panel, "ret").sort_index()
    if market_code not in ret.columns:
        raise ValueError(f"长期反转计算缺少市场指数列: {market_code}")

    alpha_rows = []
    for i in range(window - 1, len(ret)):
        block = ret.iloc[i - window + 1 : i + 1]
        m_vals = block[market_code].fillna(0).values
        sub = block.drop(columns=[market_code]).fillna(0)
        y = sub.values

        sum_w = np.sum(w)
        sum_wm = np.dot(w, m_vals)
        sum_wmm = np.dot(w, m_vals**2)
        sum_wy = w @ y
        sum_wmy = (w * m_vals) @ y

        det = sum_w * sum_wmm - sum_wm**2
        if det == 0:
            alpha_v = sum_wy / sum_w if sum_w != 0 else np.zeros(y.shape[1])
        else:
            alpha_v = (sum_wmm * sum_wy - sum_wm * sum_wmy) / det

        day = block.index[-1]
        for code, a in zip(sub.columns, alpha_v):
            alpha_rows.append({"trade_date": day, "stock_code": code, "Longterm_Alpha": -a})

    alpha_df = pd.DataFrame(alpha_rows)
    fac = rs_df.merge(alpha_df, on=["trade_date", "stock_code"], how="outer")
    fac["Long_term_reversal"] = 0.5 * fac["Longterm_Relative_strength"] + 0.5 * fac["Longterm_Alpha"]
    return fac


# -----------------------------
# Growth
# -----------------------------
def compute_growth_factor(
    asharefinancialindicator: pd.DataFrame,
    ashareeodderivativeindicator: pd.DataFrame,
    fy3: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    功能: 计算 Growth 模块（EGRO/SGRO/EGIBS）并合成 Growth。
    Input: asharefinancialindicator、ashareeodderivativeindicator、fy3。
    Output: 含 Growth 与底层成长描述词的 DataFrame。
    """
    daily_spine = _ensure_datetime(ashareeodderivativeindicator[["stock_code", "trade_date"]].copy()).sort_values("trade_date")
    eps = _ensure_datetime(asharefinancialindicator[["stock_code", "trade_date", "s_fa_eps_basic"]].copy())
    orps = _ensure_datetime(asharefinancialindicator[["stock_code", "trade_date", "s_fa_orps"]].copy())

    fy3_df = pd.DataFrame(columns=["stock_code", "trade_date", "est_eps"])
    if fy3 is not None and not fy3.empty:
        fy3_df = _ensure_datetime(fy3.copy())
        if "est_dt" in fy3_df.columns:
            fy3_df = fy3_df.rename(columns={"est_dt": "trade_date"})

    def _calc_growth(df: pd.DataFrame, value_col: str, years: int) -> pd.DataFrame:
        """
        功能: 将低频指标滚动回归得到成长率并对齐到日频。
        Input: df(原始低频指标), value_col(指标列), years(滚动年数)。
        Output: 日频成长率 DataFrame。
        """
        w = years * 12
        wide = pd.pivot_table(df, index="trade_date", columns="stock_code", values=value_col).resample("M").last().ffill()

        def slope_norm(arr: np.ndarray) -> float:
            """
            功能: 计算滚动窗口内斜率并按均值绝对值归一化。
            Input: arr(窗口内时序数组)。
            Output: 单个窗口的归一化斜率。
            """
            n = len(arr)
            if n < max(6, w // 2):
                return np.nan
            x = np.arange(n)
            mask = ~np.isnan(arr)
            if mask.sum() < max(6, w // 2):
                return np.nan
            xv, yv = x[mask], arr[mask]
            v = np.var(xv, ddof=1)
            if v == 0:
                return np.nan
            slope = np.cov(xv, yv)[0, 1] / v
            denom = np.nanmean(np.abs(yv))
            return slope / denom if denom != 0 else np.nan

        g = wide.rolling(window=w, min_periods=max(6, w // 2)).apply(lambda x: slope_norm(x), raw=True)
        long = g.reset_index().melt(id_vars="trade_date", value_name=f"{value_col}_Growth_Rate").dropna()

        long = long.sort_values("trade_date")
        merged = pd.merge_asof(
            daily_spine.sort_values("trade_date"),
            long,
            on="trade_date",
            by="stock_code",
            direction="backward",
        )
        return merged.dropna(subset=[f"{value_col}_Growth_Rate"])

    egro = _calc_growth(eps, "s_fa_eps_basic", 5)
    sgro = _calc_growth(orps, "s_fa_orps", 5)

    if fy3_df.empty or "est_eps" not in fy3_df.columns:
        egibs = pd.DataFrame(columns=["stock_code", "trade_date", "est_eps_Growth_Rate"])
    else:
        egibs = _calc_growth(fy3_df[["stock_code", "trade_date", "est_eps"]], "est_eps", 2)

    fac = egro.merge(sgro, on=["stock_code", "trade_date"], how="outer")
    fac = fac.merge(egibs, on=["stock_code", "trade_date"], how="outer")
    fac = fac.fillna(0)

    for col in ["s_fa_eps_basic_Growth_Rate", "s_fa_orps_Growth_Rate", "est_eps_Growth_Rate"]:
        if col not in fac.columns:
            fac[col] = 0.0
        fac[f"Z_{col}"] = _cross_sectional_z(fac, col)

    fac["Growth"] = 0.47 * fac["Z_est_eps_Growth_Rate"] + 0.24 * fac["Z_s_fa_orps_Growth_Rate"] + 0.29 * fac["Z_s_fa_eps_basic_Growth_Rate"]
    return fac[["trade_date", "stock_code", "Growth", "s_fa_eps_basic_Growth_Rate", "s_fa_orps_Growth_Rate", "est_eps_Growth_Rate"]]


# -----------------------------
# Sentiment A / B / C
# -----------------------------
def compute_sentiment_in_memory(
    fy0_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_obs: int = 5,
    lag: int = 63,
    horizon_n: int = 4,
) -> pd.DataFrame:
    """
    功能: Sentiment A 版本，纯内存计算 Pred_EP_chg、Pred_EPS_chg 与 Sentiment。
    Input: fy0_df(FY0 数据), start_date/end_date/min_obs/lag/horizon_n(参数)。
    Output: 含情绪描述词与 Sentiment 的 DataFrame。
    """
    if fy0_df is None or fy0_df.empty:
        raise ValueError("FY0 数据为空，无法执行 Sentiment A")

    data = _ensure_datetime(fy0_df.copy())
    if "est_dt" in data.columns:
        data = data.rename(columns={"est_dt": "trade_date"})

    needed = {"stock_code", "trade_date", "est_pe", "est_eps"}
    miss = needed - set(data.columns)
    if miss:
        raise ValueError(f"Sentiment A 缺少字段: {sorted(miss)}")

    if start_date:
        data = data[data["trade_date"] >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data["trade_date"] <= pd.to_datetime(end_date)]

    data = data.sort_values(["stock_code", "trade_date"])
    data["est_ep"] = 1.0 / data["est_pe"].replace(0, np.nan)

    def _cummean_chg(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
        """
        功能: 计算某预期序列的累计均值变化信号。
        Input: df(含 value_col 的数据), value_col(输入列), out_col(输出列名)。
        Output: 列为 trade_date/stock_code/out_col 的 DataFrame。
        """
        tmp = df[["stock_code", "trade_date", value_col]].copy()

        def _one_stock(g: pd.DataFrame) -> pd.Series:
            """
            功能: 对单股票序列计算递增样本累计均值。
            Input: g(单股票时序数据)。
            Output: 与 g 对齐的累计均值序列。
            """
            vals = g[value_col].values
            out = []
            pool = []
            for v in vals:
                pool.append(v)
                if len(pool) < min_obs:
                    out.append(np.nan)
                else:
                    out.append(np.nanmean(pool))
            return pd.Series(out, index=g.index)

        tmp[f"{value_col}_cummean"] = tmp.groupby("stock_code", group_keys=False).apply(_one_stock)

        wide = pd.pivot_table(tmp, index="trade_date", columns="stock_code", values=f"{value_col}_cummean")
        chg = wide.pct_change(periods=lag)
        parts = [chg.shift(i * lag).div(i + 1).fillna(0.0) for i in range(horizon_n)]
        comb = parts[0].copy()
        for p in parts[1:]:
            comb = comb.add(p, fill_value=0.0)
        out = comb.reset_index().melt(id_vars="trade_date", value_name=out_col).dropna()
        return out

    ep_chg = _cummean_chg(data, "est_ep", "Pred_EP_chg")
    eps_chg = _cummean_chg(data, "est_eps", "Pred_EPS_chg")

    out = ep_chg.merge(eps_chg, on=["trade_date", "stock_code"], how="outer")
    out["Sentiment"] = 0.5 * out["Pred_EP_chg"].fillna(0) + 0.5 * out["Pred_EPS_chg"].fillna(0)
    return out[["stock_code", "trade_date", "Pred_EP_chg", "Pred_EPS_chg", "Sentiment"]]


def load_sentiment_from_csv(pred_ep_csv: str, pred_eps_csv: str, strict_schema: bool = True) -> pd.DataFrame:
    """
    功能: Sentiment B 版本，从历史 CSV 读取并合成 Sentiment。
    Input: pred_ep_csv、pred_eps_csv(文件路径), strict_schema(是否严格校验字段)。
    Output: 含 Pred_EP_chg/Pred_EPS_chg/Sentiment 的 DataFrame。
    """
    a = pd.read_csv(pred_ep_csv)
    b = pd.read_csv(pred_eps_csv)

    required_a = {"stock_code", "trade_date", "Pred_EP_chg"}
    required_b = {"stock_code", "trade_date", "Pred_EPS_chg"}

    if strict_schema:
        if not required_a.issubset(a.columns):
            raise ValueError(f"Pred_EP CSV 缺少字段: {sorted(required_a - set(a.columns))}")
        if not required_b.issubset(b.columns):
            raise ValueError(f"Pred_EPS CSV 缺少字段: {sorted(required_b - set(b.columns))}")

    a["trade_date"] = pd.to_datetime(a["trade_date"], errors="coerce")
    b["trade_date"] = pd.to_datetime(b["trade_date"], errors="coerce")
    a = a.dropna(subset=["trade_date"]).drop_duplicates(["stock_code", "trade_date"], keep="last")
    b = b.dropna(subset=["trade_date"]).drop_duplicates(["stock_code", "trade_date"], keep="last")

    out = a[["stock_code", "trade_date", "Pred_EP_chg"]].merge(
        b[["stock_code", "trade_date", "Pred_EPS_chg"]],
        on=["stock_code", "trade_date"],
        how="outer",
    )
    out["Sentiment"] = 0.5 * out["Pred_EP_chg"].fillna(0) + 0.5 * out["Pred_EPS_chg"].fillna(0)
    return out


def build_sentiment(
    mode: str,
    fy0_df: Optional[pd.DataFrame] = None,
    pred_ep_csv: Optional[str] = None,
    pred_eps_csv: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    功能: Sentiment 总入口，按 A/B/C 模式执行（C 为 A 失败后回退 B）。
    Input: mode(A/B/C), fy0_df, pred_ep_csv, pred_eps_csv, kwargs(参数透传)。
    Output: Sentiment 结果 DataFrame。
    """
    mode = mode.upper()

    if mode == "A":
        if fy0_df is None:
            raise ValueError("Sentiment A 需要 fy0_df")
        return compute_sentiment_in_memory(fy0_df, **kwargs)

    if mode == "B":
        if not pred_ep_csv or not pred_eps_csv:
            raise ValueError("Sentiment B 需要 pred_ep_csv 与 pred_eps_csv")
        return load_sentiment_from_csv(pred_ep_csv, pred_eps_csv)

    if mode == "C":
        try:
            if fy0_df is None:
                raise ValueError("Sentiment C 触发 A 失败：fy0_df 缺失")
            return compute_sentiment_in_memory(fy0_df, **kwargs)
        except Exception as e:
            if not pred_ep_csv or not pred_eps_csv:
                raise RuntimeError(f"Sentiment C 模式 A 失败且未提供 CSV 回退: {e}") from e
            return load_sentiment_from_csv(pred_ep_csv, pred_eps_csv)

    raise ValueError("Sentiment mode 仅支持 A/B/C")


# -----------------------------
# Dividend
# -----------------------------
def compute_dividend_factor(
    ashareeodderivativeindicator: pd.DataFrame,
    fy1: Optional[pd.DataFrame],
    pctchange: pd.DataFrame,
) -> pd.DataFrame:
    """
    功能: 计算 Dividend 模块（历史股息率 + 分析师预期股息率）。
    Input: ashareeodderivativeindicator、fy1、pctchange。
    Output: 含 Dividend_to_price_ratio/Analyst_dp/Dividend 的 DataFrame。
    """
    base = _ensure_datetime(ashareeodderivativeindicator[["stock_code", "trade_date", "s_price_div_dps"]].copy())
    base = base.rename(columns={"s_price_div_dps": "Dividend_to_price_ratio"})

    analyst = pd.DataFrame(columns=["stock_code", "trade_date", "Analyst_dp"])
    if fy1 is not None and not fy1.empty:
        f = _ensure_datetime(fy1.copy())
        if "est_dt" in f.columns:
            f = f.rename(columns={"est_dt": "trade_date"})
        if "est_dps" in f.columns:
            px = _ensure_datetime(pctchange[["stock_code", "trade_date", "close"]].copy())
            analyst = f[["stock_code", "trade_date", "est_dps"]].merge(px, on=["stock_code", "trade_date"], how="left")
            analyst["Analyst_dp"] = analyst["est_dps"] / analyst["close"].replace(0, np.nan)
            analyst = analyst[["stock_code", "trade_date", "Analyst_dp"]]

    out = base.merge(analyst, on=["stock_code", "trade_date"], how="left")
    out["Dividend"] = 0.5 * out["Dividend_to_price_ratio"].fillna(0) + 0.5 * out["Analyst_dp"].fillna(0)
    return out


# -----------------------------
# Composite entry
# -----------------------------
def compute_all_factors(
    tables: Dict[str, pd.DataFrame],
    momentum_mode: str = "C",
    sentiment_mode: str = "C",
    sentiment_csv: Optional[Tuple[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    功能: 统一计算全部因子并按配置输出 A/B/C 模式结果。
    Input: tables(核心数据表字典), momentum_mode, sentiment_mode, sentiment_csv。
    Output: 因子名到 DataFrame 的字典。
    """
    out: Dict[str, pd.DataFrame] = {}

    out["Size"] = compute_size_factors(tables["ashareeodderivativeindicator"])
    out["Liquidity"] = compute_liquidity_factors(tables["ashareeodderivativeindicator"])

    out["Leverage"] = compute_quality_leverage(
        tables["ashareeodderivativeindicator"],
        tables["asharebalancesheet"],
        tables["asharefinancialindicator"],
    )

    out["Earnings_Quality"] = compute_quality_earnings_quality(
        tables["asharebalancesheet"],
        tables["asharecashflow"],
        tables["asharefinancialindicator"],
    )

    out["Profitability"] = compute_quality_profitability(
        tables["asharefinancialindicator"],
        tables["ashareincome"],
        tables["asharebalancesheet"],
    )

    out["Investment_Quality"] = compute_quality_investment(
        tables["asharebalancesheet"],
        tables["ashareeodderivativeindicator"],
        tables["asharecashflow"],
    )

    out["BTOP"] = compute_value_btop(tables["ashareeodderivativeindicator"])
    out["Earnings_Yield"] = compute_value_earnings_yield(
        tables["ashareeodderivativeindicator"],
        tables["asharefinancialindicator"],
        tables["asharebalancesheet"],
        tables["asharecashflow"],
        tables.get("FY0", pd.DataFrame()),
    )

    out["Growth"] = compute_growth_factor(
        tables["asharefinancialindicator"],
        tables["ashareeodderivativeindicator"],
        tables.get("FY3", pd.DataFrame()),
    )

    out["Dividend"] = compute_dividend_factor(
        tables["ashareeodderivativeindicator"],
        tables.get("FY1", pd.DataFrame()),
        tables["pctchange"],
    )

    price_stock = tables["pctchange"]
    price_index = tables["aindexeodprices"]

    out["Volatility"] = compute_volatility_factors(price_stock, price_index)
    out["Long_term_reversal"] = compute_value_long_term_reversal(price_stock, price_index)

    m_mode = momentum_mode.upper()
    if m_mode == "A":
        out["Momentum_A"] = compute_momentum_A(price_stock)
    elif m_mode == "B":
        out["Momentum_B"] = compute_momentum_B(
            price_stock,
            tables["ashareciticsindustry"],
            tables["ashareeodderivativeindicator"][["stock_code", "trade_date", "s_dq_mv"]],
        )
    elif m_mode == "C":
        risk, full = compute_momentum_C(
            price_stock,
            tables["ashareciticsindustry"],
            tables["ashareeodderivativeindicator"][["stock_code", "trade_date", "s_dq_mv"]],
        )
        out["Momentum_Portfolio"] = risk
        out["Momentum_CNE6"] = full
    else:
        raise ValueError("momentum_mode 仅支持 A/B/C")

    s_mode = sentiment_mode.upper()
    csv_a = sentiment_csv[0] if sentiment_csv else None
    csv_b = sentiment_csv[1] if sentiment_csv else None
    out["Sentiment"] = build_sentiment(
        s_mode,
        fy0_df=tables.get("FY0", pd.DataFrame()),
        pred_ep_csv=csv_a,
        pred_eps_csv=csv_b,
    )

    out["Earnings_Variability"] = compute_quality_earnings_variability(
        tables["ashareincome"],
        tables["asharecashflow"],
        tables.get("FY0", pd.DataFrame()),
        tables["pctchange"],
    )

    return out
