import pandas as pd
import pytest

from src.data_loader import validate_required_columns


def test_validate_required_columns_all_present():
    """Test that no exception is raised when all required columns are present."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]})
    required = ["col1", "col2"]

    # Should not raise an exception
    validate_required_columns(df, required, "test_table")


def test_validate_required_columns_missing_some():
    """Test that a ValueError is raised when required columns are missing."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    required = ["col1", "col3", "col4"]

    with pytest.raises(ValueError) as exc_info:
        validate_required_columns(df, required, "test_table")

    expected_missing = ["col3", "col4"]
    assert "test_table" in str(exc_info.value)
    assert str(expected_missing) in str(exc_info.value)


def test_validate_required_columns_empty_required():
    """Test that no exception is raised when required_cols is empty."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    required = []

    # Should not raise an exception
    validate_required_columns(df, required, "test_table")


def test_validate_required_columns_empty_df():
    """Test that an exception is raised when df is empty but required_cols is not."""
    df = pd.DataFrame()
    required = ["col1"]

    with pytest.raises(ValueError) as exc_info:
        validate_required_columns(df, required, "test_table")

    expected_missing = ["col1"]
    assert "test_table" in str(exc_info.value)
    assert str(expected_missing) in str(exc_info.value)
