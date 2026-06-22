import pandas as pd
import pytest

from src.common.exceptions import DataValidationError
from src.preprocessing.property_cleaner import FlatCleaner
from src.preprocessing.property_cleaner import HouseCleaner


def test_price_parser_converts_lac_and_crore_values() -> None:
    assert FlatCleaner._parse_price("90 Lac") == 0.9
    assert FlatCleaner._parse_price("1.5 Crore") == 1.5
    assert FlatCleaner._parse_price("Price on Request") is None


def test_flat_floor_parser_handles_text_floor_names() -> None:
    assert FlatCleaner._parse_floor_info("Lower Ground of 4 Floors") == (
        -0.5,
        4.0,
    )
    assert FlatCleaner._parse_floor_info("Ground of 10 Floors") == (0.0, 10.0)


def test_cleaning_stage_reports_missing_required_columns() -> None:
    cleaner = HouseCleaner()

    with pytest.raises(DataValidationError, match="price"):
        cleaner.clean_price(pd.DataFrame({"society": ["example"]}))
