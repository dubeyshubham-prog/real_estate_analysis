from pathlib import Path

import joblib

from config.settings import Config
from src.common.exceptions import EstateAIError
from src.common.exceptions import ResourceNotFoundError


class RecommendationArtifactError(EstateAIError):
    """Raised when the hybrid recommendation artifact is unavailable."""


class RecommendationService:
    """Load the recommender lazily and provide case-insensitive search."""

    def __init__(
        self,
        recommender_path: Path = Config.RECOMMENDATION_MODEL_FILE,
    ) -> None:
        self.recommender_path = Path(recommender_path)
        self._recommender = None

    @property
    def recommender(self):
        if self._recommender is None:
            if not self.recommender_path.is_file():
                raise RecommendationArtifactError(
                    "Recommendation model not found at: "
                    f"{self.recommender_path}"
                )
            self._recommender = joblib.load(self.recommender_path)
        return self._recommender

    def get_property_names(self) -> list[str]:
        return sorted(self.recommender.property_names.tolist())

    def find_property_name(self, property_name: str) -> str | None:
        normalized = property_name.strip().casefold()
        property_lookup = {
            item.casefold(): item
            for item in self.get_property_names()
        }
        return property_lookup.get(normalized)

    def get_recommendations(
        self,
        property_name: str,
        top_n: int = 5,
    ) -> list[dict[str, object]]:
        matched_property = self.find_property_name(property_name)
        if matched_property is None:
            raise ResourceNotFoundError(
                f"Property not found: {property_name}"
            )

        result = self.recommender.recommend(
            property_name=matched_property,
            top_n=top_n,
        )
        return result.to_dict(orient="records")
