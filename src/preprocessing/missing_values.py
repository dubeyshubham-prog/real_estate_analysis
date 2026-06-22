from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from config.settings import Config
from src.common.exceptions import DataValidationError
from src.features.validation import require_columns
from src.monitoring.logging import get_logger


class MissingValuePipeline:
    """Impute property fields using robust ratios and grouped business rules."""

    AREA_COLUMNS_TO_DROP: ClassVar[tuple[str, ...]] = (
        "area",
        "areaWithType",
        "super_built_up_area",
        "carpet_area",
        "area_room_ratio",
    )

    def __init__(self, missing_floor_value: float = 0.0) -> None:
        self.missing_floor_value = missing_floor_value
        self.logger = get_logger(__name__)

    @staticmethod
    def _median_positive_ratio(
        numerator: pd.Series,
        denominator: pd.Series,
        ratio_name: str,
    ) -> float:
        ratio = (numerator / denominator.replace(0, np.nan)).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        median_ratio = ratio.dropna().median()
        if pd.isna(median_ratio) or median_ratio <= 0:
            raise DataValidationError(
                f"Unable to calculate a valid {ratio_name} ratio"
            )
        return float(median_ratio)

    def impute_built_up_area(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        require_columns(
            dataframe,
            {"super_built_up_area", "built_up_area", "carpet_area"},
            "Built-up-area imputation",
        )
        imputed = dataframe.copy()
        complete = imputed.dropna(
            subset=[
                "super_built_up_area",
                "built_up_area",
                "carpet_area",
            ]
        )
        super_ratio = self._median_positive_ratio(
            complete["super_built_up_area"],
            complete["built_up_area"],
            "super-to-built-up",
        )
        carpet_ratio = self._median_positive_ratio(
            complete["carpet_area"],
            complete["built_up_area"],
            "carpet-to-built-up",
        )

        built_missing = imputed["built_up_area"].isna()
        super_present = imputed["super_built_up_area"].notna()
        carpet_present = imputed["carpet_area"].notna()

        both_mask = built_missing & super_present & carpet_present
        imputed.loc[both_mask, "built_up_area"] = (
            (
                imputed.loc[both_mask, "super_built_up_area"] / super_ratio
                + imputed.loc[both_mask, "carpet_area"] / carpet_ratio
            )
            / 2
        ).round()

        super_only_mask = built_missing & super_present & ~carpet_present
        imputed.loc[super_only_mask, "built_up_area"] = (
            imputed.loc[super_only_mask, "super_built_up_area"] / super_ratio
        ).round()

        carpet_only_mask = built_missing & ~super_present & carpet_present
        imputed.loc[carpet_only_mask, "built_up_area"] = (
            imputed.loc[carpet_only_mask, "carpet_area"] / carpet_ratio
        ).round()
        return imputed

    def drop_area_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.drop(
            columns=list(self.AREA_COLUMNS_TO_DROP),
            errors="ignore",
        ).copy()

    def impute_floor_num(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        require_columns(dataframe, {"floorNum"}, "Floor-number imputation")
        imputed = dataframe.copy()
        imputed["floorNum"] = pd.to_numeric(
            imputed["floorNum"],
            errors="coerce",
        ).fillna(self.missing_floor_value)
        return imputed

    def impute_floor_details(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Fill current and total floors using property-type-aware rules."""
        require_columns(
            dataframe,
            {
                "property_type",
                "sector",
                "floorNum",
                "floor_num",
                "total_floors",
            },
            "Floor-detail imputation",
        )
        imputed = dataframe.copy()
        for column in ("floorNum", "floor_num", "total_floors"):
            imputed[column] = pd.to_numeric(
                imputed[column],
                errors="coerce",
            )
        house_mask = imputed["property_type"].eq("house")
        imputed.loc[house_mask, "floor_num"] = (
            imputed.loc[house_mask, "floor_num"].fillna(0.0)
        )
        imputed.loc[house_mask, "total_floors"] = (
            imputed.loc[house_mask, "total_floors"]
            .fillna(imputed.loc[house_mask, "floorNum"])
        )

        flat_mask = imputed["property_type"].eq("flat")
        for column in ("floor_num", "total_floors"):
            sector_median = imputed.loc[flat_mask].groupby("sector")[
                column
            ].transform("median")
            imputed.loc[flat_mask, column] = (
                imputed.loc[flat_mask, column]
                .fillna(sector_median)
                .fillna(imputed.loc[flat_mask, column].median())
                .fillna(self.missing_floor_value)
            )
        return imputed

    @staticmethod
    def drop_facing_column(dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.drop(columns=["facing"], errors="ignore").copy()

    @staticmethod
    def drop_missing_society_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
        require_columns(
            dataframe,
            {"society"},
            "Missing-society handling",
        )
        return dataframe.loc[dataframe["society"].notna()].copy()

    @staticmethod
    def _group_mode_lookup(
        dataframe: pd.DataFrame,
        group_columns: list[str],
    ) -> pd.Series:
        known = dataframe.loc[dataframe["agePossession"].ne("Undefined")]
        return known.groupby(group_columns)["agePossession"].agg(
            lambda values: values.mode().iloc[0]
        )

    def impute_age_possession(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        require_columns(
            dataframe,
            {"agePossession", "sector", "property_type"},
            "Age-possession imputation",
        )
        imputed = dataframe.copy()
        undefined_mask = imputed["agePossession"].eq("Undefined")

        sector_type_mode = self._group_mode_lookup(
            imputed,
            ["sector", "property_type"],
        )
        sector_mode = self._group_mode_lookup(imputed, ["sector"])
        property_type_mode = self._group_mode_lookup(
            imputed,
            ["property_type"],
        )

        keys = pd.MultiIndex.from_frame(
            imputed.loc[undefined_mask, ["sector", "property_type"]]
        )
        sector_type_values = sector_type_mode.reindex(keys).to_numpy()
        imputed.loc[undefined_mask, "agePossession"] = sector_type_values

        still_missing = imputed["agePossession"].isna() | imputed[
            "agePossession"
        ].eq("Undefined")
        imputed.loc[still_missing, "agePossession"] = (
            imputed.loc[still_missing, "sector"].map(sector_mode)
        )

        still_missing = imputed["agePossession"].isna() | imputed[
            "agePossession"
        ].eq("Undefined")
        imputed.loc[still_missing, "agePossession"] = (
            imputed.loc[still_missing, "property_type"].map(property_type_mode)
        )
        return imputed

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(
            "Starting missing-value treatment; rows=%d",
            len(dataframe),
        )
        imputed = self.impute_built_up_area(dataframe)
        imputed = self.drop_area_columns(imputed)
        imputed = self.impute_floor_num(imputed)
        imputed = self.impute_floor_details(imputed)
        imputed = self.drop_facing_column(imputed)
        imputed = self.drop_missing_society_rows(imputed)
        imputed = self.impute_age_possession(imputed)
        self.logger.info(
            "Completed missing-value treatment; rows=%d columns=%d",
            *imputed.shape,
        )
        return imputed

    @staticmethod
    def save(
        dataframe: pd.DataFrame,
        output_path: Path = Config.MISSING_VALUE_TREATED_CSV,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)
        return path


MissingValueHandler = MissingValuePipeline
