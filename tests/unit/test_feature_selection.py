import pandas as pd

from src.features.selection import FeatureSelector


def test_feature_preparation_preserves_human_readable_categories() -> None:
    dataframe = pd.DataFrame(
        {
            "price": [1.0, 2.0, 3.0],
            "society": ["a", "b", "c"],
            "price_per_sqft": [1000.0, 2000.0, 3000.0],
            "luxury_score": [10.0, 45.0, 70.0],
            "floorNum": [1.0, 5.0, 12.0],
            "property_type": ["flat", "flat", "house"],
        }
    )

    result = FeatureSelector().prepare_training_data(dataframe)

    assert result["luxury_category"].tolist() == ["Low", "Medium", "High"]
    assert result["floor_category"].tolist() == [
        "Low Floor",
        "Mid Floor",
        "High Floor",
    ]
    assert result["property_type"].tolist() == ["flat", "flat", "house"]


def test_candidate_features_are_removed_without_encoding_export() -> None:
    prepared = pd.DataFrame(
        {
            "property_type": ["flat"] * 10 + ["house"] * 10,
            "bedRoom": [2.0] * 10 + [4.0] * 10,
            "pooja room": [0, 1] * 10,
            "study room": [1, 0] * 10,
            "others": [0] * 20,
            "price": [1.0] * 10 + [3.0] * 10,
        }
    )
    selector = FeatureSelector(cv_splits=2)

    result, _, _ = selector.select_final_features(prepared)

    assert "pooja room" not in result
    assert "study room" not in result
    assert "others" not in result
    assert result["property_type"].dtype == object
