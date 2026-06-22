from typing import ClassVar

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.features.validation import require_columns


class PriceModelData:
    """Prepare human-readable modelling data for training and inference."""

    TARGET_COLUMN: ClassVar[str] = "price"
    FURNISHING_LABELS: ClassVar[dict[int, str]] = {
        0: "unfurnished",
        1: "semifurnished",
        2: "furnished",
    }

    @classmethod
    def prepare_features(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        prepared = dataframe.copy()
        if "furnishing_type" in prepared.columns:
            prepared["furnishing_type"] = (
                prepared["furnishing_type"]
                .replace(cls.FURNISHING_LABELS)
                .astype("string")
            )
        return prepared

    @classmethod
    def split_features_target(
        cls,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        require_columns(
            dataframe,
            {cls.TARGET_COLUMN},
            "Price-model data preparation",
        )
        features = cls.prepare_features(
            dataframe.drop(columns=[cls.TARGET_COLUMN])
        )
        target = pd.to_numeric(
            dataframe[cls.TARGET_COLUMN],
            errors="coerce",
        )
        valid_target = target.notna() & target.gt(0)
        return features.loc[valid_target].copy(), target.loc[valid_target].copy()


def create_price_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing from the actual training feature schema."""
    categorical_columns = features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_columns = features.columns.difference(categorical_columns).tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
