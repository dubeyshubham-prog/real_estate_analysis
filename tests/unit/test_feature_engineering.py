import pandas as pd

from src.features.area_features import AreaFeatureEngineer
from src.features.furnishing_features import FurnishingClusterEngineer
from src.features.furnishing_features import FurnishingExtractor
from src.features.luxury_features import FeaturesLuxuryScoreEngineer
from src.features.property_features import AgePossessionFeatureEngineer


def test_area_engineer_converts_square_metres_to_square_feet() -> None:
    dataframe = pd.DataFrame(
        {
            "area": [900.0],
            "areaWithType": ["Carpet area: 900 (83.61 sq.m.)"],
        }
    )

    result = AreaFeatureEngineer().run(dataframe)

    assert result.loc[0, "carpet_area"] == 899.97


def test_age_categories_include_five_to_ten_year_properties() -> None:
    categorize = AgePossessionFeatureEngineer._categorize_age_possession

    assert categorize("5 to 10 Year Old") == "Moderately Old"
    assert categorize("Dec-25") == "Under Construction"


def test_furnishing_clusters_are_ordered_by_intensity() -> None:
    dataframe = pd.DataFrame(
        {
            "property_type": ["flat"] * 6,
            "AC": [0, 0, 1, 1, 4, 5],
            "Fan": [0, 1, 2, 3, 5, 6],
        }
    )

    result = FurnishingClusterEngineer().run(dataframe)

    grouped_intensity = (
        dataframe.assign(
            cluster=result["furnishing_type"],
            intensity=dataframe[["AC", "Fan"]].sum(axis=1),
        )
        .groupby("cluster")["intensity"]
        .mean()
    )
    assert grouped_intensity.is_monotonic_increasing


def test_malformed_furnishing_and_luxury_lists_are_safe() -> None:
    assert FurnishingExtractor._parse_details("not a list") == []
    assert FeaturesLuxuryScoreEngineer._parse_feature_list("not a list") == []
