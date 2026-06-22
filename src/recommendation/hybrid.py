from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.recommendation.common import validate_recommendation_data
from src.recommendation.facility import FacilityRecommender
from src.recommendation.location import LocationRecommender
from src.recommendation.price import PriceConfigRecommender


class HybridRecommender:
    """Combine explainable facility, price, and location similarities."""

    DEFAULT_WEIGHTS = {
        "facility": 0.5,
        "price": 0.3,
        "location": 0.2,
    }

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        configured = dict(weights or self.DEFAULT_WEIGHTS)
        if set(configured) != set(self.DEFAULT_WEIGHTS):
            raise ValueError(
                "Hybrid weights must contain facility, price, and location"
            )
        if any(weight < 0 for weight in configured.values()):
            raise ValueError("Hybrid weights cannot be negative")
        total = sum(configured.values())
        if total <= 0:
            raise ValueError("At least one hybrid weight must be positive")
        self.weights = {
            name: weight / total
            for name, weight in configured.items()
        }
        self.facility_recommender = FacilityRecommender()
        self.price_recommender = PriceConfigRecommender()
        self.location_recommender = LocationRecommender()

    def fit(self, dataframe: pd.DataFrame) -> "HybridRecommender":
        self.dataframe = validate_recommendation_data(dataframe)
        self.facility_recommender.fit(self.dataframe)
        self.price_recommender.fit(self.dataframe)
        self.location_recommender.fit(self.dataframe)
        self.property_names = self.location_recommender.property_names
        return self

    def _component_matrices(self) -> dict[str, np.ndarray]:
        return {
            "facility": (
                self.facility_recommender.similarity_matrix + 1
            ) / 2,
            "price": (
                self.price_recommender.similarity_matrix + 1
            ) / 2,
            "location": (
                self.location_recommender.similarity_matrix + 1
            ) / 2,
        }

    def recommend(
        self,
        property_name: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        if property_name not in self.property_names:
            raise ValueError(f"Property not found: {property_name}")

        matrices = self._component_matrices()
        hybrid = sum(
            self.weights[name] * matrix
            for name, matrix in matrices.items()
        )
        source_index = self.property_names.get_loc(property_name)
        scores = hybrid[source_index]
        candidate_indexes = np.argsort(scores)[::-1]
        candidate_indexes = candidate_indexes[
            candidate_indexes != source_index
        ][:top_n]

        return pd.DataFrame(
            {
                "PropertyName": self.property_names[candidate_indexes],
                "SimilarityScore": scores[candidate_indexes],
                "FacilityScore": matrices["facility"][
                    source_index,
                    candidate_indexes,
                ],
                "PriceScore": matrices["price"][
                    source_index,
                    candidate_indexes,
                ],
                "LocationScore": matrices["location"][
                    source_index,
                    candidate_indexes,
                ],
            }
        ).reset_index(drop=True)
