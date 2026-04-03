import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.risk_model_service import build_risk_matrices_for_date

@patch("src.risk_model_service.compute_specific_variance_matrix")
@patch("src.risk_model_service.align_specific_to_universe")
@patch("src.risk_model_service.compute_factor_covariance_matrix")
@patch("src.risk_model_service.validate_risk_inputs")
@patch("src.risk_model_service.build_daily_exposure_matrix")
def test_build_risk_matrices_for_date_happy_path(
    mock_build_exp,
    mock_validate,
    mock_compute_f,
    mock_align,
    mock_compute_delta
):
    # Setup mocks
    mock_x = pd.DataFrame(
        [[1.0, 0.5], [0.8, 1.2]],
        index=["s1", "s2"],
        columns=["f1", "f2"]
    )
    mock_build_exp.return_value = mock_x

    mock_f = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["f1", "f2"],
        columns=["f1", "f2"]
    )
    mock_compute_f.return_value = mock_f

    mock_delta = pd.DataFrame(
        [[0.05, 0.0], [0.0, 0.08]],
        index=["s1", "s2"],
        columns=["s1", "s2"]
    )
    mock_compute_delta.return_value = mock_delta

    # Setup dummy inputs
    fr = pd.DataFrame(columns=["f1", "f2"])
    sr = pd.DataFrame()
    ep = pd.DataFrame()
    tgt = "2023-01-01"

    # Call function
    res = build_risk_matrices_for_date(
        factor_returns=fr,
        specific_returns_wide=sr,
        exposure_panel=ep,
        target_date=tgt,
        factor_order=["f1", "f2"]
    )

    # Asserts
    mock_build_exp.assert_called_once()
    mock_validate.assert_called_once()
    mock_compute_f.assert_called_once()
    mock_align.assert_called_once()
    mock_compute_delta.assert_called_once()

    assert set(res.keys()) == {"X", "F", "Delta", "V", "target_date"}

    pd.testing.assert_frame_equal(res["X"], mock_x)
    pd.testing.assert_frame_equal(res["F"], mock_f)
    pd.testing.assert_frame_equal(res["Delta"], mock_delta)

    x_val = mock_x.values
    f_val = mock_f.values
    d_val = mock_delta.values
    expected_v = x_val @ f_val @ x_val.T + d_val
    expected_v = (expected_v + expected_v.T) / 2.0
    expected_v = expected_v + np.eye(2) * 1e-8
    expected_v_df = pd.DataFrame(expected_v, index=["s1", "s2"], columns=["s1", "s2"])

    pd.testing.assert_frame_equal(res["V"], expected_v_df)


@patch("src.risk_model_service.compute_specific_variance_matrix")
@patch("src.risk_model_service.align_specific_to_universe")
@patch("src.risk_model_service.compute_factor_covariance_matrix")
@patch("src.risk_model_service.validate_risk_inputs")
@patch("src.risk_model_service.build_daily_exposure_matrix")
def test_build_risk_matrices_for_date_default_factor_order(
    mock_build_exp,
    mock_validate,
    mock_compute_f,
    mock_align,
    mock_compute_delta
):
    mock_x = pd.DataFrame(
        [[1.0, 0.5]],
        index=["s1"],
        columns=["f_auto1", "f_auto2"]
    )
    mock_build_exp.return_value = mock_x
    mock_f = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]], index=["f_auto1", "f_auto2"], columns=["f_auto1", "f_auto2"])
    mock_compute_f.return_value = mock_f
    mock_compute_delta.return_value = pd.DataFrame([[0.05]], index=["s1"], columns=["s1"])

    fr = pd.DataFrame(columns=["f_auto1", "f_auto2"])

    res = build_risk_matrices_for_date(
        factor_returns=fr,
        specific_returns_wide=pd.DataFrame(),
        exposure_panel=pd.DataFrame(),
        target_date="2023-01-01"
    )

    # Check that compute_factor_covariance_matrix was called with the columns of X
    mock_compute_f.assert_called_once_with(
        factor_returns=fr,
        factor_order=["f_auto1", "f_auto2"],
        target_date=pd.to_datetime("2023-01-01"),
        lag=5,
        cov_days=252,
        mc=300,
        alpha=1.5
    )

@patch("src.risk_model_service.validate_risk_inputs")
@patch("src.risk_model_service.build_daily_exposure_matrix")
def test_build_risk_matrices_for_date_no_overlap(
    mock_build_exp,
    mock_validate
):
    mock_x = pd.DataFrame([[1.0]], index=["s1"], columns=["f_x"])
    mock_build_exp.return_value = mock_x

    fr = pd.DataFrame(columns=["f_other"])

    with pytest.raises(ValueError, match="factor_returns 与 X 无重叠因子列，无法计算 F"):
        build_risk_matrices_for_date(
            factor_returns=fr,
            specific_returns_wide=pd.DataFrame(),
            exposure_panel=pd.DataFrame(),
            target_date="2023-01-01",
            factor_order=["f_x"]
        )
