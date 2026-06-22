import pandas as pd
import pytest

from src.common.exceptions import DataValidationError
from src.preprocessing.sector_cleaner import SectorPreprocessor


def test_sector_extraction_and_alias_mapping() -> None:
    dataframe = pd.DataFrame(
        {
            "property_name": [
                "2 BHK Flat in Krishna Colony",
                "3 BHK Flat in Sector 61 Gurgaon",
            ]
        }
    )
    cleaner = SectorPreprocessor(minimum_listings=1)

    result = cleaner.clean_sector(cleaner.create_sector_column(dataframe))

    assert result["sector"].tolist() == ["sector 7", "sector 61"]


def test_rare_sector_filter_uses_configured_threshold() -> None:
    dataframe = pd.DataFrame(
        {"sector": ["sector 1", "sector 1", "sector 2"]}
    )
    cleaner = SectorPreprocessor(minimum_listings=2)

    result = cleaner.remove_rare_sectors(dataframe)

    assert result["sector"].tolist() == ["sector 1", "sector 1"]


def test_sector_extraction_rejects_invalid_property_names() -> None:
    dataframe = pd.DataFrame({"property_name": ["Unknown property"]})

    with pytest.raises(DataValidationError, match="failed for 1"):
        SectorPreprocessor().create_sector_column(dataframe)
