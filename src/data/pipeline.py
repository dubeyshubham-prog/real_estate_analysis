from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config.settings import Config
from src.data.loader import DataLoader
from src.features.pipeline import PropertyFeaturePipeline
from src.features.selection import FeatureSelector
from src.monitoring.logging import get_logger
from src.preprocessing.missing_values import MissingValuePipeline
from src.preprocessing.outliers import OutlierTreatmentPipeline
from src.preprocessing.property_cleaner import PropertyPreprocessingPipeline
from src.preprocessing.sector_cleaner import SectorPreprocessor


@dataclass(frozen=True)
class DataPipelineResult:
    merged: pd.DataFrame
    sector_cleaned: pd.DataFrame
    feature_engineered: pd.DataFrame
    outlier_treated: pd.DataFrame
    missing_value_treated: pd.DataFrame
    feature_selected: pd.DataFrame
    feature_importance: pd.DataFrame
    score_all_features: float
    score_selected_features: float


class DataPipeline:
    """Run every tabular-data stage from raw CSVs to modelling data."""

    def __init__(self, loader: DataLoader | None = None) -> None:
        self.loader = loader or DataLoader()
        self.property_preprocessor = PropertyPreprocessingPipeline()
        self.sector_preprocessor = SectorPreprocessor()
        self.feature_pipeline = PropertyFeaturePipeline()
        self.outlier_pipeline = OutlierTreatmentPipeline()
        self.missing_value_pipeline = MissingValuePipeline()
        self.feature_selector = FeatureSelector()
        self.logger = get_logger(__name__)

    def run(self) -> DataPipelineResult:
        self.logger.info("Starting complete PropLens data pipeline")
        merged = self.property_preprocessor.run(
            self.loader.load_flat_data(),
            self.loader.load_house_data(),
        )
        sector_cleaned = self.sector_preprocessor.run(merged)
        feature_engineered = self.feature_pipeline.run(
            sector_cleaned,
            self.loader.load_apartment_data(),
        )
        outlier_treated = self.outlier_pipeline.run(feature_engineered)
        missing_value_treated = self.missing_value_pipeline.run(
            outlier_treated
        )
        (
            feature_selected,
            feature_importance,
            score_all,
            score_selected,
        ) = self.feature_selector.run(missing_value_treated)
        self.logger.info("Completed complete PropLens data pipeline")
        return DataPipelineResult(
            merged=merged,
            sector_cleaned=sector_cleaned,
            feature_engineered=feature_engineered,
            outlier_treated=outlier_treated,
            missing_value_treated=missing_value_treated,
            feature_selected=feature_selected,
            feature_importance=feature_importance,
            score_all_features=score_all,
            score_selected_features=score_selected,
        )

    @staticmethod
    def save(result: DataPipelineResult) -> dict[str, Path]:
        paths = {
            "merged": Config.PROCESS_MERGE_DATA,
            "sector_cleaned": Config.GURGAON_PROPERTY_V1,
            "feature_engineered": Config.GURGAON_PROPERTY_V2,
            "outlier_treated": Config.OUTLIER_TREATED_CSV,
            "missing_value_treated": Config.MISSING_VALUE_TREATED_CSV,
            "feature_selected": Config.POST_FEATURE_SELECTION,
            "feature_importance": Config.FEATURE_IMPORTANCE_REPORT,
        }
        for name, path in paths.items():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            value = getattr(result, name)
            value.to_csv(path, index=name == "feature_importance")
        return paths
