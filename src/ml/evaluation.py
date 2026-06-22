from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.ml.preprocessing import PriceModelData
from src.ml.preprocessing import create_price_preprocessor
from src.monitoring.logging import get_logger


class ModelEvaluator:
    """Compare candidate regressors with fold-local preprocessing."""

    def __init__(
        self,
        models: Mapping[str, object] | None = None,
        cv_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.models = dict(models or self._default_models())
        self.logger = get_logger(__name__)

    def _default_models(self) -> dict[str, object]:
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(
                n_estimators=150,
                random_state=self.random_state,
                n_jobs=1,
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=150,
                random_state=self.random_state,
                n_jobs=1,
            ),
            "GradientBoosting": GradientBoostingRegressor(
                random_state=self.random_state
            ),
            "XGBoost": XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                objective="reg:squarederror",
                random_state=self.random_state,
                n_jobs=1,
            ),
        }

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        features, target = PriceModelData.split_features_target(dataframe)
        log_target = np.log1p(target)
        cv = KFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        rows: list[dict[str, float | str]] = []

        for model_name, model in self.models.items():
            self.logger.info("Evaluating model: %s", model_name)
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", create_price_preprocessor(features)),
                    ("regressor", clone(model)),
                ]
            )
            scores = cross_validate(
                pipeline,
                features,
                log_target,
                cv=cv,
                scoring={
                    "r2": "r2",
                    "mae": "neg_mean_absolute_error",
                    "rmse": "neg_root_mean_squared_error",
                },
                n_jobs=1,
            )
            rows.append(
                {
                    "model": model_name,
                    "mean_log_r2": float(scores["test_r2"].mean()),
                    "std_log_r2": float(scores["test_r2"].std()),
                    "mean_log_mae": float(-scores["test_mae"].mean()),
                    "mean_log_rmse": float(-scores["test_rmse"].mean()),
                }
            )
        return pd.DataFrame(rows).sort_values(
            "mean_log_r2",
            ascending=False,
        ).reset_index(drop=True)
