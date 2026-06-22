import re
from typing import ClassVar

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.recommendation.common import parse_literal
from src.recommendation.common import top_similar
from src.recommendation.common import validate_recommendation_data


class PriceConfigRecommender:
    """Recommend projects using available configurations, areas, and prices."""

    CONFIGURATIONS: ClassVar[tuple[str, ...]] = (
        "1 BHK",
        "2 BHK",
        "3 BHK",
        "4 BHK",
        "5 BHK",
        "6 BHK",
        "1 RK",
        "Land",
    )

    @staticmethod
    def _number_range(value: object) -> tuple[float | None, float | None]:
        numbers = [
            float(number.replace(",", ""))
            for number in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", str(value))
        ]
        if not numbers:
            return None, None
        return numbers[0], numbers[-1]

    @classmethod
    def _price_range(cls, value: object) -> tuple[float | None, float | None]:
        parts = re.split(r"\s*-\s*", str(value), maxsplit=1)
        units = [cls._price_unit(part) for part in parts]
        fallback_unit = next(
            (unit for unit in reversed(units) if unit is not None),
            None,
        )
        parsed = [
            cls._price_value(part, unit or fallback_unit)
            for part, unit in zip(parts, units, strict=True)
        ]
        if not parsed or parsed[0] is None:
            return None, None
        return parsed[0], parsed[-1]

    @staticmethod
    def _price_unit(value: object) -> str | None:
        text = str(value).casefold()
        if re.search(r"\b(?:l|lac|lakh)\b", text):
            return "lakh"
        if re.search(r"\b(?:cr|crore)\b", text):
            return "crore"
        return None

    @staticmethod
    def _price_value(
        value: object,
        unit: str | None = None,
    ) -> float | None:
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        if match is None:
            return None
        amount = float(match.group())
        if unit == "lakh":
            amount /= 100
        return amount

    def _create_feature_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for _, row in dataframe.iterrows():
            details = parse_literal(row["PriceDetails"], dict)
            features: dict[str, object] = {
                "PropertyName": row["PropertyName"]
            }
            for configuration in self.CONFIGURATIONS:
                detail = details.get(configuration, {})
                if not isinstance(detail, dict):
                    detail = {}
                area_low, area_high = self._number_range(detail.get("area", ""))
                price_low, price_high = self._price_range(
                    detail.get("price-range", "")
                )
                features[f"available_{configuration}"] = int(bool(detail))
                features[f"area_low_{configuration}"] = area_low
                features[f"area_high_{configuration}"] = area_high
                features[f"price_low_{configuration}"] = price_low
                features[f"price_high_{configuration}"] = price_high
                features[f"type_{configuration}"] = detail.get(
                    "building_type",
                    "Unknown",
                )
            rows.append(features)

        frame = pd.DataFrame(rows).set_index("PropertyName")
        categorical = frame.select_dtypes(include=["object"]).columns.tolist()
        encoded = pd.get_dummies(frame, columns=categorical, dtype=float)
        numeric = encoded.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(numeric.median()).fillna(0.0)
        return numeric

    def fit(self, dataframe: pd.DataFrame) -> "PriceConfigRecommender":
        self.dataframe = validate_recommendation_data(dataframe)
        features = self._create_feature_frame(self.dataframe)
        self.normalized_features = pd.DataFrame(
            StandardScaler().fit_transform(features),
            columns=features.columns,
            index=features.index,
        )
        self.similarity_matrix = cosine_similarity(self.normalized_features)
        self.property_names = self.normalized_features.index
        return self

    def recommend(
        self,
        property_name: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        return top_similar(
            self.property_names,
            self.similarity_matrix,
            property_name,
            top_n,
        )
