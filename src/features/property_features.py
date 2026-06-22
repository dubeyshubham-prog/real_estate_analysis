import re
from typing import ClassVar

import pandas as pd

from src.features.validation import require_columns


class AdditionalRoomFeatureEngineer:
    """Create binary indicators from additional-room descriptions."""

    ROOM_COLUMNS: ClassVar[tuple[str, ...]] = (
        "study room",
        "servant room",
        "store room",
        "pooja room",
        "others",
    )

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        require_columns(
            dataframe,
            {"additionalRoom"},
            "Additional-room feature engineering",
        )
        engineered = dataframe.copy()
        room_text = engineered["additionalRoom"].fillna("").astype(str)

        for room in self.ROOM_COLUMNS:
            engineered[room] = (
                room_text.str.contains(
                    re.escape(room),
                    case=False,
                    regex=True,
                )
                .astype("int8")
            )
        return engineered


class AgePossessionFeatureEngineer:
    """Convert raw possession-age text into stable business categories."""

    @staticmethod
    def _categorize_age_possession(value: object) -> str:
        if pd.isna(value):
            return "Undefined"

        text = str(value).strip().lower()
        if text in {"", "undefined", "nan"}:
            return "Undefined"
        if any(
            phrase in text
            for phrase in (
                "0 to 1 year old",
                "within 6 months",
                "within 3 months",
            )
        ):
            return "New Property"
        if any(
            phrase in text
            for phrase in ("1 to 5 year old", "5 to 10 year old")
        ):
            return "Moderately Old"
        if "10+ year old" in text:
            return "Old Property"
        if (
            "under construction" in text
            or text.startswith("by ")
            or re.fullmatch(r"[a-z]{3}-\d{2,4}", text)
        ):
            return "Under Construction"
        return "Undefined"

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        require_columns(
            dataframe,
            {"agePossession"},
            "Age-possession feature engineering",
        )
        engineered = dataframe.copy()
        engineered["agePossession"] = engineered["agePossession"].map(
            self._categorize_age_possession
        )
        return engineered
