from src.ml.inference import PricePredictor


class PredictionService:
    """Provide form metadata and price estimates from deployed artifacts."""

    def __init__(self, predictor: PricePredictor | None = None) -> None:
        self.predictor = predictor or PricePredictor()

    @staticmethod
    def _sorted_unique(values) -> list[object]:
        return sorted(values.dropna().unique().tolist(), key=str)

    def get_form_options(self) -> dict[str, list[object]]:
        reference_data = self.predictor.reference_data
        return {
            "property_types": self._sorted_unique(
                reference_data["property_type"]
            ),
            "sectors": self._sorted_unique(reference_data["sector"]),
            "balconies": self._sorted_unique(reference_data["balcony"]),
            "age_possession": self._sorted_unique(
                reference_data["agePossession"]
            ),
            "furnishing_types": self._sorted_unique(
                reference_data["furnishing_type"]
            ),
            "luxury_categories": self._sorted_unique(
                reference_data["luxury_category"]
            ),
            "floor_categories": self._sorted_unique(
                reference_data["floor_category"]
            ),
        }

    def predict_price(
        self,
        input_data: dict[str, object],
    ) -> dict[str, float]:
        return self.predictor.predict(input_data)
