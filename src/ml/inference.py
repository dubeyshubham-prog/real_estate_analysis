import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.settings import Config
from src.common.exceptions import EstateAIError
from src.ml.preprocessing import PriceModelData


class ModelArtifactError(EstateAIError):
    """Raised when a deployed price-model artifact is unavailable or invalid."""


class PricePredictor:
    """Load price artifacts and provide validated predictions."""

    FEATURE_LABELS = {
        "property_type": "Property type",
        "sector": "Location",
        "bedRoom": "Bedrooms",
        "bathroom": "Bathrooms",
        "balcony": "Balconies",
        "agePossession": "Property age",
        "floor_num": "Floor number",
        "total_floors": "Building height",
        "built_up_area": "Built-up area",
        "servant room": "Servant room",
        "store room": "Store room",
        "furnishing_type": "Furnishing",
        "luxury_category": "Luxury category",
        "floor_category": "Floor category",
    }

    def __init__(
        self,
        model_path: Path = Config.PRICE_MODEL_FILE,
        reference_path: Path = Config.PRICE_REFERENCE_DATA_FILE,
        metadata_path: Path = Config.PRICE_MODEL_METADATA_FILE,
    ) -> None:
        self.model_path = Path(model_path)
        self.reference_path = Path(reference_path)
        self.metadata_path = Path(metadata_path)
        self._pipeline = None
        self._reference_data = None
        self._metadata = None

    def _load(self) -> None:
        missing = [
            path
            for path in (
                self.model_path,
                self.reference_path,
                self.metadata_path,
            )
            if not path.is_file()
        ]
        if missing:
            raise ModelArtifactError(
                "Missing price-model artifacts: "
                + ", ".join(str(path) for path in missing)
            )

        self._pipeline = joblib.load(self.model_path)
        self._reference_data = joblib.load(self.reference_path)
        self._metadata = json.loads(
            self.metadata_path.read_text(encoding="utf-8")
        )

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._load()
        return self._pipeline

    @property
    def reference_data(self) -> pd.DataFrame:
        if self._reference_data is None:
            self._load()
        return self._reference_data

    @property
    def metadata(self) -> dict[str, object]:
        if self._metadata is None:
            self._load()
        return self._metadata

    def _predict_frame(self, input_frame: pd.DataFrame) -> float:
        """Return one prediction in crore from prepared model features."""
        return float(np.expm1(self.pipeline.predict(input_frame))[0])

    def _baseline_value(self, feature: str) -> object:
        """Return a representative reference value for one feature."""
        values = self.reference_data[feature].dropna()
        if values.empty:
            raise ValueError(f"No reference values available for {feature}")
        if pd.api.types.is_numeric_dtype(values):
            median = values.median()
            if pd.api.types.is_integer_dtype(values):
                return int(round(median))
            return float(median)
        return values.mode().iloc[0]

    @staticmethod
    def _plain_value(value: object) -> object:
        """Convert NumPy scalar values into template/API-safe Python values."""
        return value.item() if isinstance(value, np.generic) else value

    def explain(
        self,
        input_frame: pd.DataFrame,
        predicted_price: float,
        limit: int = 5,
    ) -> list[dict[str, object]]:
        """Explain a prediction with local, model-agnostic what-if changes."""
        explanations = []
        for feature in self.metadata["feature_columns"]:
            baseline = self._baseline_value(feature)
            comparison = input_frame.copy()
            comparison.at[comparison.index[0], feature] = baseline
            baseline_prediction = self._predict_frame(comparison)
            impact = predicted_price - baseline_prediction
            if abs(impact) < 0.01:
                continue
            explanations.append(
                {
                    "feature": feature,
                    "label": self.FEATURE_LABELS.get(feature, feature),
                    "direction": "increases" if impact > 0 else "decreases",
                    "impact_cr": round(abs(impact), 2),
                    "input_value": self._plain_value(
                        input_frame.iloc[0][feature]
                    ),
                    "baseline_value": self._plain_value(baseline),
                }
            )
        explanations.sort(
            key=lambda item: float(item["impact_cr"]),
            reverse=True,
        )
        return explanations[:limit]

    def predict(self, input_data: dict[str, object]) -> dict[str, object]:
        input_frame = PriceModelData.prepare_features(
            pd.DataFrame([input_data])
        )
        expected_columns = self.metadata["feature_columns"]
        missing_columns = set(expected_columns).difference(input_frame.columns)
        if missing_columns:
            raise ValueError(
                "Missing prediction fields: "
                + ", ".join(sorted(missing_columns))
            )

        input_frame = input_frame.loc[:, expected_columns]
        predicted_price = self._predict_frame(input_frame)
        metrics = self.metadata["metrics"]
        lower = max(
            0.0,
            predicted_price + float(metrics["residual_q10_crore"]),
        )
        upper = max(
            lower,
            predicted_price + float(metrics["residual_q90_crore"]),
        )
        return {
            "predicted_price_cr": round(predicted_price, 2),
            "lower_estimate_cr": round(lower, 2),
            "upper_estimate_cr": round(upper, 2),
            "explanations": self.explain(input_frame, predicted_price),
        }
