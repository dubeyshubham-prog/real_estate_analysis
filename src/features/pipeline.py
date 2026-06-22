from pathlib import Path
from typing import ClassVar

import pandas as pd

from config.settings import Config
from src.features.area_features import AreaFeatureEngineer
from src.features.furnishing_features import FurnishingClusterEngineer
from src.features.furnishing_features import FurnishingExtractor
from src.features.luxury_features import FeaturesLuxuryScoreEngineer
from src.features.property_features import AdditionalRoomFeatureEngineer
from src.features.property_features import AgePossessionFeatureEngineer
from src.monitoring.logging import get_logger


class FinalFeatureCleaner:
    """Remove source columns after their useful information is engineered."""

    DROP_COLUMNS: ClassVar[tuple[str, ...]] = (
        "nearbyLocations",
        "furnishDetails",
        "features",
        "features_list",
        "additionalRoom",
        "address",
    )

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.drop(
            columns=list(self.DROP_COLUMNS),
            errors="ignore",
        ).copy()


class PropertyFeaturePipeline:
    """Execute the complete first-level property feature workflow."""

    def __init__(self) -> None:
        self.area_engineer = AreaFeatureEngineer()
        self.room_engineer = AdditionalRoomFeatureEngineer()
        self.age_engineer = AgePossessionFeatureEngineer()
        self.furnishing_extractor = FurnishingExtractor()
        self.furnishing_clusterer = FurnishingClusterEngineer()
        self.luxury_engineer = FeaturesLuxuryScoreEngineer()
        self.final_cleaner = FinalFeatureCleaner()
        self.logger = get_logger(__name__)

    def run(
        self,
        dataframe: pd.DataFrame,
        apartment_data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.logger.info(
            "Starting property feature engineering; rows=%d",
            len(dataframe),
        )
        engineered = self.area_engineer.run(dataframe)
        engineered = self.room_engineer.run(engineered)
        engineered = self.age_engineer.run(engineered)
        engineered = self.furnishing_extractor.run(engineered)
        engineered = self.furnishing_clusterer.run(engineered)
        engineered = self.luxury_engineer.run(engineered, apartment_data)
        engineered = self.final_cleaner.run(engineered)
        self.logger.info(
            "Completed property feature engineering; rows=%d columns=%d",
            *engineered.shape,
        )
        return engineered

    @staticmethod
    def save(
        dataframe: pd.DataFrame,
        output_path: Path = Config.GURGAON_PROPERTY_V2,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)
        return path
