import json
from datetime import datetime
from datetime import timezone
from pathlib import Path

import joblib
import pandas as pd

from config.settings import Config
from src.monitoring.logging import get_logger
from src.recommendation.hybrid import HybridRecommender


class RecommendationTrainer:
    """Train and persist the hybrid recommendation artifact."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.weights = weights
        self.logger = get_logger(__name__)

    def train(self, apartment_data: pd.DataFrame) -> HybridRecommender:
        recommender = HybridRecommender(weights=self.weights).fit(
            apartment_data
        )
        self.logger.info(
            "Recommendation model trained with %d projects",
            len(recommender.property_names),
        )
        return recommender

    @staticmethod
    def save(
        recommender: HybridRecommender,
        model_path: Path = Config.RECOMMENDATION_MODEL_FILE,
        metadata_path: Path = Config.RECOMMENDATION_METADATA_FILE,
    ) -> dict[str, Path]:
        model = Path(model_path)
        metadata = Path(metadata_path)
        model.parent.mkdir(parents=True, exist_ok=True)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(recommender, model)
        metadata.write_text(
            json.dumps(
                {
                    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                    "project_count": len(recommender.property_names),
                    "weights": recommender.weights,
                    "components": ["facility", "price", "location"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"model": model, "metadata": metadata}
