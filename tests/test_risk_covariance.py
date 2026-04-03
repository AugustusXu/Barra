import numpy as np
import pytest

from src.risk_covariance import _exp_weights

def test_exp_weights_basic():
    """Test standard valid inputs for _exp_weights."""
    length = 20
    half_life = 5

    weights = _exp_weights(length, half_life)

    # Check length
    assert len(weights) == length

    # Check sum is 1.0
    np.testing.assert_allclose(np.sum(weights), 1.0)

    # Check half-life property
    # The most recent observation is at index -1 (which has power 0).
    # The observation `half_life` periods ago is at index -1 - half_life.
    # Its weight should be exactly half of the most recent observation's weight.
    np.testing.assert_allclose(weights[-1 - half_life], 0.5 * weights[-1])

def test_exp_weights_half_life_property_multiple():
    """Test half-life property for multiple half-lives."""
    length = 60
    half_life = 10

    weights = _exp_weights(length, half_life)

    # 1 half-life ago = 0.5 weight
    np.testing.assert_allclose(weights[-1 - half_life], 0.5 * weights[-1])
    # 2 half-lives ago = 0.25 weight
    np.testing.assert_allclose(weights[-1 - 2 * half_life], 0.25 * weights[-1])
    # 3 half-lives ago = 0.125 weight
    np.testing.assert_allclose(weights[-1 - 3 * half_life], 0.125 * weights[-1])

def test_exp_weights_zero_length():
    """Test edge case with zero or negative length."""
    weights_zero = _exp_weights(0, 5)
    assert len(weights_zero) == 0
    assert isinstance(weights_zero, np.ndarray)

    weights_neg = _exp_weights(-5, 5)
    assert len(weights_neg) == 0
    assert isinstance(weights_neg, np.ndarray)

def test_exp_weights_invalid_half_life():
    """Test edge case with half_life <= 0, which is treated as 1."""
    length = 5

    # half_life = 0 should be treated as half_life = 1
    weights_zero_hl = _exp_weights(length, 0)
    weights_one_hl = _exp_weights(length, 1)

    np.testing.assert_array_equal(weights_zero_hl, weights_one_hl)

    # half_life < 0 should also be treated as half_life = 1
    weights_neg_hl = _exp_weights(length, -10)
    np.testing.assert_array_equal(weights_neg_hl, weights_one_hl)

def test_exp_weights_sum_zero_raw():
    """Test edge case when the sum of raw weights is extremely small or zero."""
    # This might happen if length is very large and half_life is 1, causing underflow.
    # We can mock or find a case where the sum might theoretically be 0,
    # but the function guards against s <= 0.

    # Since it's hard to trigger sum <= 0 naturally with double precision,
    # we can verify that the code would theoretically divide equally if sum <= 0.
    # However, testing with a very large length and small half_life will at least test stability.
    length = 2000
    half_life = 1
    weights = _exp_weights(length, half_life)

    assert len(weights) == length
    np.testing.assert_allclose(np.sum(weights), 1.0)
