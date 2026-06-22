import re
from typing import ClassVar

import pandas as pd

from src.features.validation import require_columns


class AreaFeatureEngineer:
    """Extract standardized area measurements from listing text."""

    SQM_TO_SQFT: ClassVar[float] = 10.7639

    @staticmethod
    def _extract_area(text: object, label: str) -> float | None:
        if pd.isna(text):
            return None

        match = re.search(
            rf"{re.escape(label)}\s*:?\s*(\d+(?:\.\d+)?)",
            str(text),
            flags=re.IGNORECASE,
        )
        return float(match.group(1)) if match else None

    def _super_built_up_area(self, text: object) -> float | None:
        return self._extract_area(text, "Super Built up area")

    def _get_area(self, text: object, area_type: str) -> float | None:
        return self._extract_area(text, area_type)

    def _convert_to_sqft(
        self,
        text: object,
        area_value: float | None,
    ) -> float | None:
        if area_value is None or pd.isna(text):
            return area_value

        match = re.search(
            rf"{area_value:g}\s*\((\d+(?:\.\d+)?)\s*sq\.?m\.?\)",
            str(text),
            flags=re.IGNORECASE,
        )
        if match is None:
            return area_value
        return round(float(match.group(1)) * self.SQM_TO_SQFT, 2)

    def _extract_plot_area(self, text: object) -> float | None:
        return self._extract_area(text, "Plot area")

    @staticmethod
    def _convert_scale(row: pd.Series) -> float | None:
        if pd.isna(row["area"]) or pd.isna(row["built_up_area"]):
            return row["built_up_area"]

        ratio = round(row["area"] / row["built_up_area"])
        if ratio == 9:
            return row["built_up_area"] * 9
        if ratio == 11:
            return row["built_up_area"] * 10.7
        return row["built_up_area"]

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        require_columns(
            dataframe,
            {"area", "areaWithType"},
            "Area feature engineering",
        )
        engineered = dataframe.copy()

        area_definitions = {
            "super_built_up_area": "Super Built up area",
            "built_up_area": "Built Up area",
            "carpet_area": "Carpet area",
        }
        for column_name, label in area_definitions.items():
            engineered[column_name] = engineered["areaWithType"].map(
                lambda text, area_label=label: self._extract_area(
                    text,
                    area_label,
                )
            )
            engineered[column_name] = engineered.apply(
                lambda row, output_column=column_name: self._convert_to_sqft(
                    row["areaWithType"],
                    row[output_column],
                ),
                axis=1,
            )

        missing_area_mask = engineered[
            ["super_built_up_area", "built_up_area", "carpet_area"]
        ].isna().all(axis=1)
        fallback = engineered.loc[
            missing_area_mask,
            ["area", "areaWithType", "built_up_area"],
        ].copy()
        fallback["built_up_area"] = fallback["areaWithType"].map(
            self._extract_plot_area
        )
        fallback["built_up_area"] = fallback.apply(
            self._convert_scale,
            axis=1,
        )
        engineered.update(fallback)
        return engineered
