# Barra 因子研究与风险模型

本项目将多因子研究流程从 Notebook 逐步模块化，覆盖：
- 因子计算与评估
- 因子收益率与特异收益估计（v1/v2）
- 风险模型矩阵构建（$V=XFX^\top+\Delta$）
- 调仓日风险快照批量落盘
- 组合优化与风险归因（最小可用版）

当前主入口为 `main.ipynb`，核心逻辑已拆分为多个 `.py` 模块，便于复用、调试与扩展。

---

## 1. 项目结构

```text
FactorInv/
├─ .gitignore
├─ .vscode/
├─ main.ipynb
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ __init__.py
│  ├─ data_loader.py
│  ├─ data_processor.py
│  ├─ factor_comp.py
│  ├─ factor_eval.py
│  ├─ factor_return.py
│  ├─ factor_return_v2.py
│  ├─ risk_covariance.py
│  ├─ risk_specific.py
│  ├─ risk_model_service.py
│  ├─ portfolio_optimizer.py
│  ├─ risk_attribution.py
│  └─ __pycache__/
├─ output/
└─ __pycache__/
```

---

## 2. 各文件说明

### Notebook 与文档
- `main.ipynb`：主编排 Notebook。按单元顺序执行，串联数据加载、因子计算、回测、因子收益、风险矩阵、优化与归因。
- `README.md`：当前项目总说明（本文件）。

### 数据与预处理
- `src/data_loader.py`
  - 数据源读取与校验：`load_core_tables`、`validate_required_columns`
  - 因子落盘加载：`save_by_trade_date`、`load_saved_factor_outputs`
  - 风险输入读取：`load_pctchange_daily_pkl`、`load_factor_returns`、`load_specific_returns`、`load_exposure_panel`
- `src/data_processor.py`
  - 去极值标准化：`mad_winsorize_series`、`remove_outliers_and_zscore`
  - 标签构造：`add_next_return_label`
  - 风险对齐：`align_specific_returns_to_exposure`、`preprocess_exposure_cross_section`

### 因子计算与评估
- `src/factor_comp.py`
  - 规模、波动、流动性、动量（A/B/C）、质量、价值、成长、情绪、分红等因子计算
  - 汇总入口：`compute_all_factors`
- `src/factor_eval.py`
  - IC/ICIR、分组净值、多空表现、可视化
  - 入口：`evaluate_factor`、`plot_ic_curve`、`plot_group_nav`

### 因子收益率估计
- `src/factor_return.py`
  - 基础版截面回归流程，输出因子收益率与特异收益
  - 入口：`run_factor_return_pipeline`
- `src/factor_return_v2.py`
  - 改进版（更稳健）：行业项、标准化、截距、权重方案等增强
  - 入口：`run_factor_return_pipeline_v2`

### 风险模型
- `src/risk_covariance.py`
  - 因子协方差矩阵 $F$ 估计：Newey-West、自适应调整、VRA
  - 入口：`compute_factor_covariance_matrix`
- `src/risk_specific.py`
  - 特异方差矩阵 $\Delta$ 估计
  - 入口：`compute_specific_variance_matrix`、`to_specific_returns_wide`
- `src/risk_model_service.py`
  - 服务层封装：构建单日与批量风险快照
  - 单日入口：`build_risk_matrices_for_date`
  - 批量入口：`build_risk_snapshots_for_rebalance_dates`

### 优化与归因
- `src/portfolio_optimizer.py`
  - 统一优化入口：`optimize_portfolio`
  - 支持策略：`min_abs_risk`、`min_active_risk`、`max_abs_return`、`max_active_return`
  - 支持约束：个股上限、主动权重上限、行业偏离、Size 偏离、换手/交易成本
- `src/risk_attribution.py`
  - 主动风险分解：风格/行业/特异贡献及占比
  - 入口：`attribute_portfolio_risk`

### 其他
- `.gitignore`：忽略规则。
- `.vscode/`：本地编辑器配置。
- `output/`：运行输出目录（因子、收益、风险快照等）。
- `requirements.txt`：依赖清单。

---

## 3. 环境依赖

`requirements.txt` 当前包含：
- `pandas`
- `numpy`
- `scipy`
- `statsmodels`
- `matplotlib`
- `scikit-learn`
- `joblib`
- `rich`
- `cvxpy`

---

## 4. 快速开始

1) 安装依赖（建议虚拟环境）：

```powershell
pip install -r requirements.txt
```

2) 打开并运行 `main.ipynb`：
- 先在配置区设置 `BASE_DIR`（数据根目录）
- 视需要调整：
  - `ENABLE_FACTOR_FILTER` / `TARGET_FACTORS`
  - `ENABLE_DATE_FILTER` / `BT_START` / `BT_END`
  - `BT_BUFFER_DAYS`

3) 按单元顺序执行，典型流程为：
- 加载表数据
- 计算并落盘因子
- 读取 output 并做评估/可视化
- 计算因子收益率（建议 v2）
- 构建单日风险矩阵与批量风险快照
- 运行最小优化与风险归因示例

---

## 5. 输入与输出约定

### 关键输入
- 历史行情与财务等基础表（在 `main.ipynb` 的 `TABLE_SOURCES` 中配置）
- `pctchange` 收益数据（支持按日 `pkl` 目录优先，CSV 回退）

### 关键输出（默认在 `./output`）
- 因子按日文件（各因子目录）
- 因子收益率与特异收益（`factor_return` / `factor_return_v2` 输出）
- 风险快照目录：`./output/risk_snapshots`（仅 `pkl`）

---

## 6. 模型关系（简述）

- 总风险矩阵：
$$
V = XFX^\top + \Delta
$$
其中：
- $X$：股票-因子暴露矩阵
- $F$：因子协方差矩阵
- $\Delta$：特异方差对角矩阵

组合优化与归因模块均基于上述风险表示。

---

