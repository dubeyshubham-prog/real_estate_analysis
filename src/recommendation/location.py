import re

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.recommendation.common import parse_literal
from src.recommendation.common import top_similar
from src.recommendation.common import validate_recommendation_data


class LocationRecommender:
    """Recommend projects using distances to nearby landmarks."""

    @staticmethod
    def _distance_to_meters(value: object) -> float | None:
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        if match is None:
            return None
        distance = float(match.group())
        text = str(value).casefold()
        return distance * 1000 if "km" in text else distance

    def _create_location_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        rows: dict[str, dict[str, float | None]] = {}
        for _, row in dataframe.iterrows():
            advantages = parse_literal(row["LocationAdvantages"], dict)
            rows[str(row["PropertyName"])] = {
                str(location): self._distance_to_meters(distance)
                for location, distance in advantages.items()
            }
        frame = pd.DataFrame.from_dict(rows, orient="index")

        # A missing landmark means "not advertised/unknown", so use a
        # conservative column-specific far distance instead of a magic constant.
        column_fallback = frame.max(axis=0).mul(1.25).fillna(1.0)
        return frame.fillna(column_fallback).fillna(1.0)

    def fit(self, dataframe: pd.DataFrame) -> "LocationRecommender":
        self.dataframe = validate_recommendation_data(dataframe)
        features = self._create_location_frame(self.dataframe)
        self.normalized_features = pd.DataFrame(
            StandardScaler().fit_transform(features),
            columns=features.columns,
            index=features.index,
        )
        self.similarity_matrix = cosine_similarity(self.normalized_features)
        self.property_names = self.normalized_features.index
        return self

    def recommend(
        self,
        property_name: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        return top_similar(
            self.property_names,
            self.similarity_matrix,
            property_name,
            top_n,
        )
