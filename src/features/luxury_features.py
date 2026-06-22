import ast
from collections.abc import Mapping
from typing import ClassVar

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src.features.validation import require_columns


class FeaturesLuxuryScoreEngineer:
    """Fill facility data and calculate a weighted luxury score."""

    DEFAULT_WEIGHTS: ClassVar[dict[str, int]] = {
        "24/7 Power Backup": 8,
        "24/7 Water Supply": 4,
        "24x7 Security": 7,
        "ATM": 4,
        "Aerobics Centre": 6,
        "Airy Rooms": 8,
        "Amphitheatre": 7,
        "Badminton Court": 7,
        "Banquet Hall": 8,
        "Bar/Chill-Out Lounge": 9,
        "Barbecue": 7,
        "Basketball Court": 7,
        "Business Lounge": 9,
        "CCTV Camera Security": 8,
        "Car Parking": 6,
        "Club House": 9,
        "Concierge Service": 9,
        "Fire Fighting Systems": 8,
        "Fitness Centre / GYM": 8,
        "Flower Garden": 7,
        "Gated Community": 7,
        "Golf Course": 10,
        "Gymnasium": 8,
        "High Speed Elevators": 8,
        "Intercom Facility": 7,
        "Jogging Track": 7,
        "Landscape Garden": 8,
        "Maintenance Staff": 6,
        "Park": 8,
        "Piped Gas": 7,
        "Power Back up Lift": 8,
        "Private Garden / Terrace": 9,
        "Security Personnel": 9,
        "Shopping Centre": 7,
        "Swimming Pool": 8,
        "Visitor Parking": 7,
        "Water Storage": 7,
        "Yoga/Meditation Area": 7,
    }

    def __init__(self, weights: Mapping[str, int] | None = None) -> None:
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)

    def fill_missing_features(
        self,
        dataframe: pd.DataFrame,
        apartment_data: pd.DataFrame,
    ) -> pd.DataFrame:
        require_columns(
            dataframe,
            {"features", "society"},
            "Missing luxury-feature enrichment",
        )
        require_columns(
            apartment_data,
            {"PropertyName", "TopFacilities"},
            "Apartment facility enrichment",
        )
        engineered = dataframe.copy()
        lookup = (
            apartment_data.assign(
                PropertyName=(
                    apartment_data["PropertyName"]
                    .astype("string")
                    .str.strip()
                    .str.lower()
                )
            )
            .drop_duplicates("PropertyName")
            .set_index("PropertyName")["TopFacilities"]
        )
        missing_mask = engineered["features"].isna()
        engineered.loc[missing_mask, "features"] = (
            engineered.loc[missing_mask, "society"]
            .astype("string")
            .str.strip()
            .str.lower()
            .map(lookup)
        )
        return engineered

    @staticmethod
    def _parse_feature_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if not isinstance(value, str) or not value.strip().startswith("["):
            return []
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
        return [str(item) for item in parsed] if isinstance(parsed, list) else []

    def create_features_list(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        require_columns(
            dataframe,
            {"features"},
            "Luxury feature parsing",
        )
        engineered = dataframe.copy()
        engineered["features_list"] = engineered["features"].map(
            self._parse_feature_list
        )
        return engineered

    def create_luxury_score(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        require_columns(
            dataframe,
            {"features_list"},
            "Luxury score creation",
        )
        engineered = dataframe.copy()
        binarizer = MultiLabelBinarizer()
        feature_matrix = binarizer.fit_transform(engineered["features_list"])
        binary_features = pd.DataFrame(
            feature_matrix,
            columns=binarizer.classes_,
            index=engineered.index,
        )
        available_weights = pd.Series(
            {
                feature: weight
                for feature, weight in self.weights.items()
                if feature in binary_features.columns
            },
            dtype=float,
        )
        if available_weights.empty:
            engineered["luxury_score"] = 0.0
        else:
            engineered["luxury_score"] = binary_features[
                available_weights.index
            ].mul(available_weights, axis="columns").sum(axis=1)
        return engineered

    def run(
        self,
        dataframe: pd.DataFrame,
        apartment_data: pd.DataFrame,
    ) -> pd.DataFrame:
        engineered = self.fill_missing_features(dataframe, apartment_data)
        engineered = self.create_features_list(engineered)
        return self.create_luxury_score(engineered)
