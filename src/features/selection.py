from pathlib import Path
from typing import ClassVar

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

from config.settings import Config
from src.features.validation import require_columns
from src.monitoring.logging import get_logger


class FeatureSelector:
    """Evaluate feature utility while preserving raw categories for modelling."""

    CANDIDATE_DROP_COLUMNS: ClassVar[tuple[str, ...]] = (
        "pooja room",
        "study room",
        "others",
    )

    def __init__(self, random_state: int = 42, cv_splits: int = 5) -> None:
        self.random_state = random_state
        self.cv_splits = cv_splits
        self.logger = get_logger(__name__)

    @staticmethod
    def _categorize_luxury(score: object) -> str:
        if pd.isna(score):
            return "Unknown"
        value = float(score)
        if value < 30:
            return "Low"
        if value < 60:
            return "Medium"
        return "High"

    @staticmethod
    def _categorize_floor(floor: object) -> str:
        if pd.isna(floor):
            return "Unknown"
        value = float(floor)
        if value <= 2:
            return "Low Floor"
        if value <= 10:
            return "Mid Floor"
        return "High Floor"

    def prepare_training_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        require_columns(
            dataframe,
            {"price", "luxury_score", "floorNum"},
            "Feature-selection preparation",
        )
        prepared = dataframe.drop(
            columns=["society", "price_per_sqft"],
            errors="ignore",
        ).copy()
        prepared["luxury_category"] = prepared["luxury_score"].map(
            self._categorize_luxury
        )
        prepared["floor_category"] = prepared["floorNum"].map(
            self._categorize_floor
        )
        return prepared.drop(
            columns=["floorNum", "luxury_score"],
            errors="ignore",
        )

    @staticmethod
    def _build_model_pipeline(
        features: pd.DataFrame,
        model,
    ) -> Pipeline:
        categorical_columns = features.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
        numeric_columns = features.columns.difference(
            categorical_columns
        ).tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    SimpleImputer(strategy="median"),
                    numeric_columns,
                ),
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(strategy="most_frequent"),
                            ),
                            (
                                "encoder",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                ),
                            ),
                        ]
                    ),
                    categorical_columns,
                ),
            ],
            remainder="drop",
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    def label_encode_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Create a diagnostic encoded copy without changing export data."""
        encoded = dataframe.copy()
        categorical_columns = encoded.select_dtypes(
            include=["object", "string", "category"]
        ).columns
        numeric_columns = encoded.columns.difference(categorical_columns)

        for column in numeric_columns:
            encoded[column] = encoded[column].fillna(
                encoded[column].median()
            )
        if len(categorical_columns):
            for column in categorical_columns:
                mode = encoded[column].mode()
                fill_value = mode.iloc[0] if not mode.empty else "Unknown"
                encoded[column] = encoded[column].fillna(fill_value)
            encoded[categorical_columns] = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ).fit_transform(encoded[categorical_columns])
        return encoded

    def create_feature_importance_report(
        self,
        prepared_data: pd.DataFrame,
    ) -> pd.DataFrame:
        encoded = self.label_encode_data(prepared_data)
        features = encoded.drop(columns=["price"])
        target = encoded["price"]

        random_forest = RandomForestRegressor(
            n_estimators=150,
            random_state=self.random_state,
            n_jobs=1,
        )
        gradient_boosting = GradientBoostingRegressor(
            random_state=self.random_state
        )
        random_forest.fit(features, target)
        gradient_boosting.fit(features, target)

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=self.random_state,
        )
        permutation_model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=1,
        ).fit(x_train, y_train)
        permutation = permutation_importance(
            permutation_model,
            x_test,
            y_test,
            n_repeats=5,
            random_state=self.random_state,
            n_jobs=1,
        )

        return (
            pd.DataFrame(
                {
                    "feature": features.columns,
                    "correlation": features.corrwith(target).values,
                    "rf_importance": random_forest.feature_importances_,
                    "gb_importance": gradient_boosting.feature_importances_,
                    "permutation_importance": permutation.importances_mean,
                }
            )
            .set_index("feature")
            .sort_values("permutation_importance", ascending=False)
        )

    def _cross_validated_score(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> float:
        pipeline = self._build_model_pipeline(
            features,
            RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=1,
            ),
        )
        cv = KFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        return float(
            cross_val_score(
                pipeline,
                features,
                target,
                cv=cv,
                scoring="r2",
                n_jobs=1,
            ).mean()
        )

    def select_final_features(
        self,
        prepared_data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, float, float]:
        features = prepared_data.drop(columns=["price"])
        target = prepared_data["price"]
        score_all = self._cross_validated_score(features, target)

        selected_features = features.drop(
            columns=list(self.CANDIDATE_DROP_COLUMNS),
            errors="ignore",
        )
        score_selected = self._cross_validated_score(
            selected_features,
            target,
        )
        export_data = selected_features.copy()
        export_data["price"] = target
        return export_data, score_all, score_selected

    def run(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
        self.logger.info(
            "Starting feature selection; rows=%d",
            len(dataframe),
        )
        prepared = self.prepare_training_data(dataframe)
        report = self.create_feature_importance_report(prepared)
        export_data, score_all, score_selected = self.select_final_features(
            prepared
        )
        self.logger.info(
            "Completed feature selection; features=%d score_all=%.4f "
            "score_selected=%.4f",
            export_data.shape[1] - 1,
            score_all,
            score_selected,
        )
        return export_data, report, score_all, score_selected

    @staticmethod
    def save(
        dataframe: pd.DataFrame,
        report: pd.DataFrame,
        output_path: Path = Config.POST_FEATURE_SELECTION,
        report_path: Path = Config.FEATURE_IMPORTANCE_REPORT,
    ) -> tuple[Path, Path]:
        output = Path(output_path)
        feature_report = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        feature_report.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output, index=False)
        report.to_csv(feature_report)
        return output, feature_report
