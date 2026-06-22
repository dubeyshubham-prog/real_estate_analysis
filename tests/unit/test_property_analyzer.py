import pandas as pd
import pytest

from src.analysis.property_analyzer import PropertyAnalyzer
from src.common.exceptions import DataValidationError


@pytest.fixture
def property_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "property_type": ["flat", "house", "flat"],
            "sector": ["sector 1", "sector 2", "sector 1"],
            "price": [1.0, 3.0, 2.0],
            "built_up_area": [1000.0, 2000.0, 1500.0],
            "bedRoom": [2.0, 4.0, 3.0],
            "luxury_category": ["Low", "High", "Medium"],
            "furnishing_type": [0.0, 2.0, 1.0],
        }
    )


def test_market_summary_aggregates_expected_values(
    property_data: pd.DataFrame,
) -> None:
    summary = PropertyAnalyzer().market_summary(property_data, top_n=2)

    assert summary["total_properties"] == 3
    assert summary["average_price_cr"] == 2.0
    assert summary["property_type_distribution"] == {"flat": 2, "house": 1}
    assert summary["top_expensive_sectors"] == {
        "sector 2": 3.0,
        "sector 1": 1.5,
    }


def test_prepare_geo_data_parses_noisy_degree_symbols(
    property_data: pd.DataFrame,
) -> None:
    coordinates = pd.DataFrame(
        {
            "sector": ["sector 1", "sector 2"],
            "coordinates": [
                "28.3663� N, 76.9456� E",
                "28.5095° N, 77.0320° E",
            ],
        }
    )

    result = PropertyAnalyzer().prepare_geo_data(
        property_data,
        coordinates,
    )

    parsed_coordinates = (
        result.groupby("sector")[["latitude", "longitude"]]
        .first()
        .to_dict(orient="index")
    )
    assert parsed_coordinates == {
        "sector 1": {"latitude": 28.3663, "longitude": 76.9456},
        "sector 2": {"latitude": 28.5095, "longitude": 77.032},
    }


def test_analysis_reports_missing_columns() -> None:
    with pytest.raises(DataValidationError, match="price"):
        PropertyAnalyzer().market_summary(
            pd.DataFrame({"sector": ["sector 1"]})
        )
