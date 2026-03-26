from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, cast

import pandas as pd


DEFAULT_DATE_COLS = ("trade_date", "ann_dt", "est_dt", "report_period")



def ensure_datetime_columns(df: pd.DataFrame, date_cols: Iterable[str] = DEFAULT_DATE_COLS) -> pd.DataFrame:
    """
    功能: 将指定日期列统一转换为 pandas datetime 类型。
    Input: df(原始表), date_cols(需要转换的列名集合)。
    Output: 转换后的 DataFrame 副本。
    """
    result = df.copy()
    for col in date_cols:
        if col in result.columns:
            result[col] = pd.to_datetime(result[col], errors="coerce")
    return result


def validate_required_columns(df: pd.DataFrame, required_cols: Iterable[str], name: str) -> None:
    """
    功能: 校验数据表是否包含必需字段。
    Input: df(待校验表), required_cols(必需列), name(表名用于报错信息)。
    Output: 无返回；缺列时抛出 ValueError。
    """
    required = set(required_cols)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} 缺少必要字段: {missing}")


def load_all_pkl_files(directory: str | Path) -> List[pd.DataFrame]:
    """
    功能: 批量读取目录下所有 pkl 文件并返回 DataFrame 列表。
    Input: directory(pkl 文件目录路径)。
    Output: List[pd.DataFrame]，按文件名排序后的结果。
    """
    path = Path(directory)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"目录不存在: {path}")

    files = sorted(path.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"目录下未找到 pkl 文件: {path}")

    frames = []
    for file in files:
        frames.append(pd.read_pickle(file))
    return frames


def load_table_from_source(source: str | Path, source_type: str = "auto") -> pd.DataFrame:
    """
    功能: 从目录/CSV/PKL 单源加载表，并做日期列标准化。
    Input: source(路径), source_type(auto|pkl_dir|csv|pkl)。
    Output: 读取并处理后的 DataFrame。
    """
    src = Path(source)
    st = source_type.lower()

    if st == "auto":
        if src.is_dir():
            st = "pkl_dir"
        elif src.suffix.lower() == ".csv":
            st = "csv"
        elif src.suffix.lower() in {".pkl", ".pickle"}:
            st = "pkl"
        else:
            raise ValueError(f"无法自动识别数据源类型: {src}")

    if st == "pkl_dir":
        df = pd.concat(load_all_pkl_files(src), ignore_index=True)
    elif st == "csv":
        df = pd.read_csv(src)
    elif st == "pkl":
        df = pd.read_pickle(src)
    else:
        raise ValueError(f"不支持的数据源类型: {source_type}")

    return ensure_datetime_columns(df)


def load_core_tables(table_sources: Mapping[str, Mapping[str, str]]) -> Dict[str, pd.DataFrame]:
    """
    功能: 按配置加载核心数据表，支持重命名与必需列校验。
    Input: table_sources(每张表的 path/type/rename/required 配置字典)。
    Output: Dict[str, pd.DataFrame]，键为表名。
    """
    tables: Dict[str, pd.DataFrame] = {}
    for table_name, conf in table_sources.items():
        if "path" not in conf:
            raise ValueError(f"{table_name} 缺少 path 配置")
        source_type = conf.get("type", "auto")
        table = load_table_from_source(conf["path"], source_type=source_type)

        rename_map = cast(dict, conf.get("rename", {}))
        if rename_map:
            table = table.rename(columns=rename_map)

        required = conf.get("required", [])
        if required:
            validate_required_columns(table, required, table_name)

        tables[table_name] = table
    return tables


def save_by_trade_date(df: pd.DataFrame, folder_path: str | Path, date_col: str = "trade_date") -> Path:
    """
    功能: 按交易日拆分 DataFrame 并逐日保存为 pkl。
    Input: df(待保存数据), folder_path(输出目录), date_col(日期列名)。
    Output: 最后一个写入文件的 Path。
    """
    if date_col not in df.columns:
        raise ValueError(f"数据不包含日期列: {date_col}")

    out_dir = Path(folder_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])

    last_file: Optional[Path] = None
    data["_save_date"] = data[date_col].apply(lambda x: x.date().isoformat())
    for date, group in data.groupby("_save_date"):
        file_path = out_dir / f"{date}.pkl"
        group.drop(columns=["_save_date"], errors="ignore").to_pickle(file_path)
        last_file = file_path

    if last_file is None:
        raise ValueError("没有可保存的数据（日期列全部为空或数据为空）")
    return last_file


def load_saved_factor_outputs(output_dir_map, target_factors=None, start_date=None, end_date=None):
    """
    功能: 从指定目录加载已保存的因子输出，支持按因子名过滤和日期范围过滤。
    Input: output_dir_map(因子名到目录路径的映射), target_factors(要加载的因子列表), start_date(起始日期), end_date(结束日期)。
    Output: Dict[str, pd.DataFrame]，键为因子名，值为对应的 DataFrame。
     - 仅加载 target_factors 中指定的因子（如果提供）。
    """
    target_set = set(target_factors) if target_factors else None
    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None

    loaded = {}
    for factor_name, folder in output_dir_map.items():
        if target_set is not None and factor_name not in target_set:
            continue

        folder = Path(folder)
        if not folder.exists():
            continue

        pkl_files = sorted(folder.glob('*.pkl'))
        if not pkl_files:
            continue

        frames = []
        for fp in pkl_files:
            # 如果文件名是日期（例如 2024-01-31.pkl），先做文件级过滤，减少读取量
            stem_dt = pd.to_datetime(fp.stem, errors='coerce')
            if pd.notna(stem_dt):
                if start_dt is not None and stem_dt < start_dt:
                    continue
                if end_dt is not None and stem_dt > end_dt:
                    continue
            try:
                df_part = pd.read_pickle(fp)
            except Exception:
                continue

            if {'stock_code', 'trade_date'}.issubset(df_part.columns):
                df_part['trade_date'] = pd.to_datetime(df_part['trade_date'], errors='coerce')
                df_part = df_part.dropna(subset=['trade_date'])
                if start_dt is not None:
                    df_part = df_part[df_part['trade_date'] >= start_dt]
                if end_dt is not None:
                    df_part = df_part[df_part['trade_date'] <= end_dt]
                if df_part.empty:
                    continue
            frames.append(df_part)

        if not frames:
            continue

        df = pd.concat(frames, ignore_index=True)
        if {'stock_code', 'trade_date'}.issubset(df.columns):
            df = df.drop_duplicates(['stock_code', 'trade_date'], keep='last')
        loaded[factor_name] = df

    return loaded


def _to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def load_pctchange_daily_pkl(
    pctchange_daily_dir: str | Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    buffer_days: int = 0,
) -> pd.DataFrame:
    """
    功能: 从按交易日切分的 pkl 目录读取收益数据，并按日期窗口过滤。
    Input: pctchange_daily_dir(目录), start_date/end_date(可选), buffer_days(前后扩展天数)。
    Output: 含 stock_code/trade_date/pctchange 的 DataFrame。
    """
    folder = Path(pctchange_daily_dir)
    if not folder.exists():
        raise FileNotFoundError(f"目录不存在: {folder}")

    start_dt = pd.to_datetime(start_date, errors="coerce") if start_date else None
    end_dt = pd.to_datetime(end_date, errors="coerce") if end_date else None

    if start_dt is not None and pd.isna(start_dt):
        start_dt = None
    if end_dt is not None and pd.isna(end_dt):
        end_dt = None

    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        raise ValueError(f"start_date 不能晚于 end_date: {start_date} > {end_date}")

    if start_dt is not None:
        start_dt = start_dt - pd.Timedelta(days=buffer_days)
    if end_dt is not None:
        end_dt = end_dt + pd.Timedelta(days=buffer_days)

    selected_files: List[Path] = []
    for fp in sorted(folder.glob("*.pkl")):
        dt = pd.to_datetime(fp.stem, errors="coerce")
        if pd.isna(dt):
            continue
        if start_dt is not None and dt < start_dt:
            continue
        if end_dt is not None and dt > end_dt:
            continue
        selected_files.append(fp)

    frames: List[pd.DataFrame] = []
    for fp in selected_files:
        try:
            part = pd.read_pickle(fp)
        except Exception:
            continue

        cols = [c for c in ["stock_code", "trade_date", "pctchange"] if c in part.columns]
        if len(cols) < 3:
            continue
        part = part[cols].copy()
        part["trade_date"] = _to_datetime_safe(part["trade_date"])
        part = part.dropna(subset=["trade_date"])
        frames.append(part)

    if not frames:
        return pd.DataFrame(columns=["stock_code", "trade_date", "pctchange"])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["stock_code", "trade_date"], keep="last")
    return out


def load_exposure_panel(path: str | Path) -> pd.DataFrame:
    """
    功能: 读取暴露面板并规范关键键列格式。
    Input: path(Exposure_Matrix_Panel 的 pkl 路径)。
    Output: 暴露面板 DataFrame。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"暴露面板文件不存在: {p}")

    panel = pd.read_pickle(p)
    if "trade_date" not in panel.columns:
        raise ValueError("暴露面板缺少 trade_date 列")
    if "stock_code" not in panel.columns:
        raise ValueError("暴露面板缺少 stock_code 列")

    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel = panel.dropna(subset=["trade_date"])
    panel = panel.drop_duplicates(["stock_code", "trade_date"], keep="last")
    return panel


def load_factor_returns(path: str | Path, index_col: int | str = 0) -> pd.DataFrame:
    """
    功能: 读取因子收益序列（csv/pkl），并将索引转换为日期。
    Input: path(文件路径), index_col(索引列配置)。
    Output: index=trade_date 的因子收益 DataFrame。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"因子收益文件不存在: {p}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p, index_col=index_col)
    elif p.suffix.lower() in {".pkl", ".pickle"}:
        df = pd.read_pickle(p)
    else:
        raise ValueError(f"不支持的因子收益文件类型: {p.suffix}")

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()
    return df


def load_specific_returns(
    path: str | Path,
    stock_col: str = "stock_code",
    date_col: str = "trade_date",
    value_col: str = "specific_return",
) -> pd.DataFrame:
    """
    功能: 读取特异性收益（支持长表/宽表），统一为宽表(index=stock, columns=date)。
    Input: path(文件路径), stock_col/date_col/value_col(长表字段名)。
    Output: 宽表特异性收益 DataFrame。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"特异性收益文件不存在: {p}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() in {".pkl", ".pickle"}:
        df = pd.read_pickle(p)
    else:
        raise ValueError(f"不支持的特异性收益文件类型: {p.suffix}")

    if {stock_col, date_col, value_col}.issubset(df.columns):
        long_df = df[[stock_col, date_col, value_col]].copy()
        long_df[date_col] = pd.to_datetime(long_df[date_col], errors="coerce")
        long_df = long_df.dropna(subset=[date_col])
        wide = long_df.pivot_table(index=stock_col, columns=date_col, values=value_col, aggfunc="last")
        wide = wide.sort_index().sort_index(axis=1)
        return wide

    if df.index.name is None and stock_col in df.columns:
        df = df.set_index(stock_col)

    try:
        df.columns = pd.to_datetime(df.columns, errors="coerce")
        df = df.loc[:, ~df.columns.isna()]
        return df.sort_index().sort_index(axis=1)
    except Exception as exc:
        raise ValueError("无法识别特异性收益数据结构，请提供长表或日期列宽表") from exc