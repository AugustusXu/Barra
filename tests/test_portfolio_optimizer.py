from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from src.portfolio_optimizer import ExtendedOptimizeConfig, build_trading_constraints

def test_build_trading_constraints_base_case():
    """Test constraints builder without previous weights."""
    n = 5
    weights = cp.Variable(n)
    cfg = ExtendedOptimizeConfig(single_stock_cap=0.1, enforce_long_only=True)

    constraints, turnover_expr, cost_expr = build_trading_constraints(
        weights, prev_weights=None, cfg=cfg
    )

    # Check length: enforce_long_only (1) + sum=1 (1) + single_stock_cap (1) = 3
    assert len(constraints) == 3

    # Check that turnover and cost expressions are Constants equal to 0.0
    assert isinstance(turnover_expr, cp.Constant)
    assert turnover_expr.value == 0.0

    assert isinstance(cost_expr, cp.Constant)
    assert cost_expr.value == 0.0


def test_build_trading_constraints_with_prev_weights():
    """Test constraints builder with previous weights and default turnover caps."""
    n = 5
    weights = cp.Variable(n)
    prev_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    cfg = ExtendedOptimizeConfig(max_turnover=0.5) # max_turnover < 1.0 adds 1 constraint

    constraints, turnover_expr, cost_expr = build_trading_constraints(
        weights, prev_weights=prev_weights, cfg=cfg
    )

    # enforce_long_only (1) + sum=1 (1) + single_stock_cap (1) + max_turnover (1) = 4
    assert len(constraints) == 4

    # Turnover and cost should be expression now, not constant
    assert not isinstance(turnover_expr, cp.Constant)
    assert not isinstance(cost_expr, cp.Constant)


def test_build_trading_constraints_buy_sell_caps():
    """Test constraints builder with buy and sell turnover caps."""
    n = 5
    weights = cp.Variable(n)
    prev_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    cfg = ExtendedOptimizeConfig(
        max_turnover=1.5, # > 1.0, so no max_turnover constraint
        buy_turnover_cap=0.3,
        sell_turnover_cap=0.4
    )

    constraints, turnover_expr, cost_expr = build_trading_constraints(
        weights, prev_weights=prev_weights, cfg=cfg
    )

    # enforce_long_only (1) + sum=1 (1) + single_stock_cap (1) + buy_cap (1) + sell_cap (1) = 5
    assert len(constraints) == 5


def test_build_trading_constraints_no_long_only():
    """Test constraints builder when enforce_long_only is False."""
    n = 5
    weights = cp.Variable(n)
    cfg = ExtendedOptimizeConfig(enforce_long_only=False)

    constraints, turnover_expr, cost_expr = build_trading_constraints(
        weights, prev_weights=None, cfg=cfg
    )

    # sum=1 (1) + single_stock_cap (1) = 2
    assert len(constraints) == 2
