import pandas as pd

from src.preprocessing.missing_values import MissingValuePipeline


def test_age_imputation_uses_sector_and_property_type_mode() -> None:
    dataframe = pd.DataFrame(
        {
            "sector": ["sector 1", "sector 1", "sector 1"],
            "property_type": ["flat", "flat", "house"],
            "agePossession": [
                "New Property",
                "Undefined",
                "Old Property",
            ],
        }
    )

    result = MissingValuePipeline().impute_age_possession(dataframe)

    assert result.loc[1, "agePossession"] == "New Property"


def test_missing_floor_uses_explicit_default() -> None:
    dataframe = pd.DataFrame({"floorNum": [None, 4.0]})

    result = MissingValuePipeline(missing_floor_value=0.0).impute_floor_num(
        dataframe
    )

    assert result["floorNum"].tolist() == [0.0, 4.0]


def test_house_floor_details_use_ground_and_total_floor_count() -> None:
    dataframe = pd.DataFrame(
        {
            "property_type": ["house"],
            "sector": ["sector 1"],
            "floorNum": [3.0],
            "floor_num": [None],
            "total_floors": [None],
        }
    )

    result = MissingValuePipeline().impute_floor_details(dataframe)

    assert result.loc[0, "floor_num"] == 0.0
    assert result.loc[0, "total_floors"] == 3.0
