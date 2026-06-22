import json
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from config.settings import Config
from src.ml.preprocessing import PriceModelData
from src.ml.preprocessing import create_price_preprocessor
from src.monitoring.logging import get_logger


@dataclass(frozen=True)
class TrainingMetrics:
    r2: float
    mae_crore: float
    rmse_crore: float
    median_absolute_error_crore: float
    residual_q10_crore: float
    residual_q90_crore: float


@dataclass(frozen=True)
class TrainingResult:
    pipeline: Pipeline
    reference_data: pd.DataFrame
    metrics: TrainingMetrics
    metadata: dict[str, object]


class PriceModelTrainer:
    """Train, evaluate, and persist the production price pipeline."""

    DEFAULT_PARAMS = {
        "learning_rate": 0.1,
        "max_depth": 5,
        "n_estimators": 250,
        "subsample": 0.8,
    }

    def __init__(
        self,
        model_params: dict[str, object] | None = None,
        random_state: int = 42,
        test_size: float = 0.2,
    ) -> None:
        self.model_params = {
            **self.DEFAULT_PARAMS,
            **(model_params or {}),
        }
        self.random_state = random_state
        self.test_size = test_size
        self.logger = get_logger(__name__)

    def create_pipeline(self, features: pd.DataFrame) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocessor", create_price_preprocessor(features)),
                (
                    "regressor",
                    XGBRegressor(
                        **self.model_params,
                        objective="reg:squarederror",
                        random_state=self.random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    def run(self, dataframe: pd.DataFrame) -> TrainingResult:
        features, target = PriceModelData.split_features_target(dataframe)
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        pipeline = self.create_pipeline(features)
        pipeline.fit(x_train, np.log1p(y_train))

        predicted_price = np.expm1(pipeline.predict(x_test))
        residuals = y_test.to_numpy() - predicted_price
        absolute_errors = np.abs(residuals)
        metrics = TrainingMetrics(
            r2=float(r2_score(y_test, predicted_price)),
            mae_crore=float(mean_absolute_error(y_test, predicted_price)),
            rmse_crore=float(
                mean_squared_error(y_test, predicted_price) ** 0.5
            ),
            median_absolute_error_crore=float(np.median(absolute_errors)),
            residual_q10_crore=float(np.quantile(residuals, 0.10)),
            residual_q90_crore=float(np.quantile(residuals, 0.90)),
        )

        # Refit the deployable pipeline on all available validated data.
        pipeline.fit(features, np.log1p(target))
        metadata: dict[str, object] = {
            "model_type": "XGBRegressor",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_rows": len(features),
            "feature_columns": features.columns.tolist(),
            "model_params": self.model_params,
            "metrics": asdict(metrics),
        }
        self.logger.info(
            "Price model trained; rows=%d holdout_r2=%.4f mae=%.4f crore",
            len(features),
            metrics.r2,
            metrics.mae_crore,
        )
        return TrainingResult(
            pipeline=pipeline,
            reference_data=features,
            metrics=metrics,
            metadata=metadata,
        )

    @staticmethod
    def save(
        result: TrainingResult,
        model_path: Path = Config.PRICE_MODEL_FILE,
        reference_path: Path = Config.PRICE_REFERENCE_DATA_FILE,
        metadata_path: Path = Config.PRICE_MODEL_METADATA_FILE,
    ) -> dict[str, Path]:
        paths = {
            "model": Path(model_path),
            "reference_data": Path(reference_path),
            "metadata": Path(metadata_path),
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(result.pipeline, paths["model"])
        joblib.dump(result.reference_data, paths["reference_data"])
        paths["metadata"].write_text(
            json.dumps(result.metadata, indent=2),
            encoding="utf-8",
        )
        return paths


FinalModelTrainer = PriceModelTrainer
