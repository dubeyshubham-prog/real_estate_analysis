from pathlib import Path

import joblib
import pandas as pd

from app.services.recommendation_service import RecommendationService
from src.recommendation.common import validate_recommendation_data
from src.recommendation.hybrid import HybridRecommender
from src.recommendation.location import LocationRecommender
from src.recommendation.price import PriceConfigRecommender


def sample_apartments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PropertyName": ["Alpha", "Beta", "Gamma", "PropertyName"],
            "TopFacilities": [
                "['Pool', 'Gym']",
                "['Pool', 'Park']",
                "['School', 'Park']",
                "TopFacilities",
            ],
            "PriceDetails": [
                "{'2 BHK': {'building_type': 'Apartment', "
                "'area': '1000 sq.ft.', 'price-range': '₹ 1 - 1.2 Cr'}}",
                "{'2 BHK': {'building_type': 'Apartment', "
                "'area': '1100 sq.ft.', 'price-range': '₹ 1.1 - 1.3 Cr'}}",
                "{'3 BHK': {'building_type': 'Apartment', "
                "'area': '1800 sq.ft.', 'price-range': '₹ 2 - 2.4 Cr'}}",
                "PriceDetails",
            ],
            "LocationAdvantages": [
                "{'School': '1 KM', 'Metro': '2 KM'}",
                "{'School': '1.2 KM', 'Metro': '2.2 KM'}",
                "{'School': '5 KM', 'Metro': '6 KM'}",
                "LocationAdvantages",
            ],
        }
    )


def test_invalid_header_row_is_removed_by_content() -> None:
    valid = validate_recommendation_data(sample_apartments())

    assert valid["PropertyName"].tolist() == ["Alpha", "Beta", "Gamma"]


def test_price_and_distance_parsers_handle_real_formats() -> None:
    assert PriceConfigRecommender._price_range("₹ 90 L - 1.2 Cr") == (
        0.9,
        1.2,
    )
    assert PriceConfigRecommender._price_range("₹ 90 - 95 L") == (
        0.9,
        0.95,
    )
    assert LocationRecommender._distance_to_meters("2.5 KM") == 2500
    assert LocationRecommender._distance_to_meters("450 Meter") == 450


def test_hybrid_recommendation_excludes_source_and_explains_scores() -> None:
    recommender = HybridRecommender().fit(sample_apartments())

    result = recommender.recommend("Alpha", top_n=2)

    assert "Alpha" not in result["PropertyName"].tolist()
    assert {
        "FacilityScore",
        "PriceScore",
        "LocationScore",
    }.issubset(result.columns)
    score_columns = [
        "SimilarityScore",
        "FacilityScore",
        "PriceScore",
        "LocationScore",
    ]
    assert result[score_columns].apply(
        lambda column: column.between(0, 1).all()
    ).all()


def test_saved_recommender_supports_case_insensitive_service(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "recommender.joblib"
    joblib.dump(HybridRecommender().fit(sample_apartments()), model_path)
    service = RecommendationService(model_path)

    result = service.get_recommendations("alpha", top_n=1)

    assert len(result) == 1
    assert result[0]["PropertyName"] != "Alpha"
