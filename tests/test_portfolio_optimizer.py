import pytest
import numpy as np
import pandas as pd

from src.portfolio_optimizer import optimize_portfolio, OptimizeConfig

@pytest.fixture
def mock_data():
    """Provides a minimal, valid dataset for testing portfolio optimization."""
    # 5 assets
    assets = ['A', 'B', 'C', 'D', 'E']

    # Simple exposure matrix with Size and one Industry
    exposure_matrix = pd.DataFrame({
        'Size': [1.0, 0.5, -0.5, -1.0, 0.0],
        'Ind_Tech': [1, 1, 0, 0, 0],
        'Ind_Fin': [0, 0, 1, 1, 1]
    }, index=assets)

    # Covariance/Risk matrix - identity for simplicity
    total_risk_matrix = pd.DataFrame(np.eye(5), index=assets, columns=assets)

    # Equal weight benchmark
    benchmark_weights = pd.Series(0.2, index=assets)

    # Alpha signal
    alpha_signal = pd.Series({
        'A': 0.05,
        'B': 0.02,
        'C': -0.01,
        'D': -0.03,
        'E': 0.0
    })

    return {
        'exposure_matrix': exposure_matrix,
        'total_risk_matrix': total_risk_matrix,
        'benchmark_weights': benchmark_weights,
        'alpha_signal': alpha_signal,
    }

def test_optimize_portfolio_min_abs_risk(mock_data):
    """Test min_abs_risk strategy."""
    cfg = OptimizeConfig(single_stock_cap=1.0)

    res = optimize_portfolio(
        strategy_type="min_abs_risk",
        exposure_matrix=mock_data['exposure_matrix'],
        total_risk_matrix=mock_data['total_risk_matrix'],
        benchmark_weights=mock_data['benchmark_weights'],
        config=cfg
    )

    assert res['status'] == 'optimal'
    weights = res['weights']
    assert np.isclose(weights.sum(), 1.0)
    assert all(weights >= 0)
    assert len(weights) == 5

def test_optimize_portfolio_min_active_risk(mock_data):
    """Test min_active_risk strategy."""
    cfg = OptimizeConfig(single_stock_cap=1.0, active_weight_cap=0.5)

    res = optimize_portfolio(
        strategy_type="min_active_risk",
        exposure_matrix=mock_data['exposure_matrix'],
        total_risk_matrix=mock_data['total_risk_matrix'],
        benchmark_weights=mock_data['benchmark_weights'],
        config=cfg
    )

    assert res['status'] == 'optimal'
    weights = res['weights']
    active_weights = res['active_weights']
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(active_weights.sum(), 0.0, atol=1e-5)
    assert all(weights >= 0)

def test_optimize_portfolio_max_abs_return(mock_data):
    """Test max_abs_return strategy."""
    cfg = OptimizeConfig(single_stock_cap=1.0, risk_aversion=0.1)

    res = optimize_portfolio(
        strategy_type="max_abs_return",
        exposure_matrix=mock_data['exposure_matrix'],
        total_risk_matrix=mock_data['total_risk_matrix'],
        benchmark_weights=mock_data['benchmark_weights'],
        alpha_signal=mock_data['alpha_signal'],
        config=cfg
    )

    assert res['status'] == 'optimal'
    weights = res['weights']
    assert np.isclose(weights.sum(), 1.0)
    assert all(weights >= -1e-6)  # Account for minor floating point inaccuracies

    # Asset A has highest alpha, should have higher weight than benchmark
    assert weights['A'] > 0.2

def test_optimize_portfolio_max_active_return(mock_data):
    """Test max_active_return strategy."""
    # Using larger risk_aversion to bound the problem and ensure feasibility
    cfg = OptimizeConfig(single_stock_cap=1.0, active_weight_cap=0.5, risk_aversion=10.0, size_restriction_max_active_return=1.0, industry_dev_limit=1.0)

    res = optimize_portfolio(
        strategy_type="max_active_return",
        exposure_matrix=mock_data['exposure_matrix'],
        total_risk_matrix=mock_data['total_risk_matrix'],
        benchmark_weights=mock_data['benchmark_weights'],
        alpha_signal=mock_data['alpha_signal'],
        config=cfg
    )

    assert res['status'] == 'optimal'
    weights = res['weights']
    active_weights = res['active_weights']

    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(active_weights.sum(), 0.0, atol=1e-5)

    # Asset A has highest alpha, should have positive active weight
    assert active_weights['A'] > 0.0

def test_optimize_portfolio_empty_exposure():
    """Test that empty exposure matrix raises ValueError."""
    with pytest.raises(ValueError, match="exposure_matrix 为空"):
        optimize_portfolio(
            strategy_type="min_abs_risk",
            exposure_matrix=pd.DataFrame(),
            total_risk_matrix=pd.DataFrame()
        )

def test_optimize_portfolio_unsupported_strategy(mock_data):
    """Test that unsupported strategy raises ValueError."""
    with pytest.raises(ValueError, match="不支持的 strategy_type: unsupported_strategy"):
        optimize_portfolio(
            strategy_type="unsupported_strategy",
            exposure_matrix=mock_data['exposure_matrix'],
            total_risk_matrix=mock_data['total_risk_matrix']
        )
