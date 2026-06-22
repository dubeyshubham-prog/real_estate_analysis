from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from config.settings import Config
from src.features.validation import require_columns
from src.monitoring.logging import get_logger


class OutlierTreatmentPipeline:
    """Apply validated statistical and domain-based property outlier rules."""

    MANUAL_DROP_INDEXES: ClassVar[tuple[int, ...]] = (
        2679,
        3488,
        181,
        3430,
        1144,
        1324,
        3071,
        2513,
        853,
        589,
        495,
        545,
        1474,
        1842,
        3061,
        1838,
        3135,
        1103,
    )

    def __init__(
        self,
        maximum_price_per_sqft: float = 50_000,
        maximum_area: float = 100_000,
        maximum_bedrooms: float = 10,
        minimum_area_per_room: float = 100,
    ) -> None:
        self.maximum_price_per_sqft = maximum_price_per_sqft
        self.maximum_area = maximum_area
        self.maximum_bedrooms = maximum_bedrooms
        self.minimum_area_per_room = minimum_area_per_room
        self.logger = get_logger(__name__)

    @staticmethod
    def _safe_price_per_sqft(dataframe: pd.DataFrame) -> pd.Series:
        area = pd.to_numeric(dataframe["area"], errors="coerce").replace(0, np.nan)
        price = pd.to_numeric(dataframe["price"], errors="coerce")
        return ((price * 10_000_000) / area).round()

    def treat_price_per_sqft_outliers(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        require_columns(
            dataframe,
            {"price", "price_per_sqft", "area"},
            "Price-per-square-foot outlier treatment",
        )
        treated = dataframe.copy()
        q1 = treated["price_per_sqft"].quantile(0.25)
        q3 = treated["price_per_sqft"].quantile(0.75)
        iqr = q3 - q1
        outlier_mask = ~treated["price_per_sqft"].between(
            q1 - 1.5 * iqr,
            q3 + 1.5 * iqr,
        )

        small_area_mask = outlier_mask & treated["area"].lt(1000)
        treated.loc[small_area_mask, "area"] *= 9
        treated.loc[outlier_mask, "price_per_sqft"] = (
            self._safe_price_per_sqft(treated.loc[outlier_mask])
        )
        return treated.loc[
            treated["price_per_sqft"].le(self.maximum_price_per_sqft)
            & treated["price_per_sqft"].notna()
        ].copy()

    def treat_area_outliers(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        require_columns(dataframe, {"area"}, "Area outlier treatment")
        treated = dataframe.loc[
            pd.to_numeric(dataframe["area"], errors="coerce").lt(
                self.maximum_area
            )
        ].copy()

        existing_manual_indexes = treated.index.intersection(
            self.MANUAL_DROP_INDEXES
        )
        if len(existing_manual_indexes):
            self.logger.warning(
                "Dropping %d legacy index-based area corrections; replace these "
                "with stable listing identifiers when source IDs are retained",
                len(existing_manual_indexes),
            )
            treated = treated.drop(index=existing_manual_indexes)
        return treated

    def treat_bedroom_outliers(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        require_columns(dataframe, {"bedRoom"}, "Bedroom outlier treatment")
        bedrooms = pd.to_numeric(dataframe["bedRoom"], errors="coerce")
        return dataframe.loc[
            bedrooms.between(1, self.maximum_bedrooms)
        ].copy()

    def recalculate_price_per_sqft(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        require_columns(
            dataframe,
            {"price", "area"},
            "Price-per-square-foot recalculation",
        )
        treated = dataframe.copy()
        treated["price_per_sqft"] = self._safe_price_per_sqft(treated)
        return treated.loc[treated["price_per_sqft"].notna()].copy()

    def treat_area_room_ratio_outliers(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        require_columns(
            dataframe,
            {"area", "bedRoom", "floorNum"},
            "Area-room-ratio treatment",
        )
        treated = dataframe.copy()
        treated["area"] = pd.to_numeric(treated["area"], errors="coerce")
        treated["bedRoom"] = pd.to_numeric(treated["bedRoom"], errors="coerce")
        treated["floorNum"] = pd.to_numeric(
            treated["floorNum"],
            errors="coerce",
        )
        treated["area_room_ratio"] = treated["area"] / treated["bedRoom"]
        treated = treated.loc[
            treated["area_room_ratio"].gt(self.minimum_area_per_room)
        ].copy()

        multi_floor_house_mask = (
            treated["area_room_ratio"].lt(250)
            & treated["bedRoom"].gt(3)
            & treated["floorNum"].notna()
            & treated["floorNum"].ne(0)
        )
        treated.loc[multi_floor_house_mask, "bedRoom"] = (
            treated.loc[multi_floor_house_mask, "bedRoom"]
            / treated.loc[multi_floor_house_mask, "floorNum"]
        ).round()
        treated["area_room_ratio"] = treated["area"] / treated["bedRoom"]

        invalid_dense_property = (
            treated["area_room_ratio"].lt(250)
            & treated["bedRoom"].gt(4)
        )
        return treated.loc[~invalid_dense_property].copy()

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Starting outlier treatment; rows=%d", len(dataframe))
        treated = dataframe.copy().reset_index(drop=True)
        treated = self.treat_price_per_sqft_outliers(treated)
        treated = self.treat_area_outliers(treated)
        treated = self.treat_bedroom_outliers(treated)
        treated = self.recalculate_price_per_sqft(treated)
        treated = self.treat_area_room_ratio_outliers(treated)
        self.logger.info(
            "Completed outlier treatment; rows=%d columns=%d",
            *treated.shape,
        )
        return treated

    @staticmethod
    def save(
        dataframe: pd.DataFrame,
        output_path: Path = Config.OUTLIER_TREATED_CSV,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)
        return path


OutlierDetector = OutlierTreatmentPipeline
