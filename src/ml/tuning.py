from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.ml.preprocessing import PriceModelData
from src.ml.preprocessing import create_price_preprocessor
from src.monitoring.logging import get_logger


class HyperparameterTuner:
    """Tune the selected price model without preprocessing leakage."""

    DEFAULT_GRID = {
        "regressor__n_estimators": [150, 250],
        "regressor__max_depth": [3, 5],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__subsample": [0.8, 1.0],
    }

    def __init__(
        self,
        parameter_grid: Mapping[str, list[object]] | None = None,
        cv_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.parameter_grid = dict(parameter_grid or self.DEFAULT_GRID)
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.logger = get_logger(__name__)

    def run(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[Pipeline, pd.DataFrame]:
        features, target = PriceModelData.split_features_target(dataframe)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_price_preprocessor(features)),
                (
                    "regressor",
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=self.random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=self.parameter_grid,
            scoring="r2",
            cv=KFold(
                n_splits=self.cv_splits,
                shuffle=True,
                random_state=self.random_state,
            ),
            n_jobs=1,
            return_train_score=False,
        )
        self.logger.info(
            "Starting XGBoost tuning with %d candidates",
            len(list(ParameterGrid(self.parameter_grid))),
        )
        search.fit(features, np.log1p(target))

        report = (
            pd.DataFrame(search.cv_results_)
            .loc[
                :,
                [
                    "rank_test_score",
                    "mean_test_score",
                    "std_test_score",
                    "params",
                ],
            ]
            .sort_values("rank_test_score")
            .reset_index(drop=True)
        )
        report = report.rename(
            columns={
                "mean_test_score": "mean_test_log_r2",
                "std_test_score": "std_test_log_r2",
            }
        )
        return search.best_estimator_, report
