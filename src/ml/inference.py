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

    def predict(self, input_data: dict[str, object]) -> dict[str, float]:
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
        predicted_price = float(
            np.expm1(self.pipeline.predict(input_frame))[0]
        )
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
        }
