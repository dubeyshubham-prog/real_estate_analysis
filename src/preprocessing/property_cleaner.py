import re
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar
import pandas as pd

from config.settings import Config
from src.common.exceptions import DataValidationError
from src.monitoring.logging import get_logger


class BasePropertyCleaner:
    """Provide shared validation, parsing, and cleaning operations."""

    DROP_COLUMNS: ClassVar[tuple[str, ...]] = ("link", "property_id")
    RATING_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\d+(?:\.\d+)?\s?(?:★|â˜…)"
    )
    NUMBER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\d+(?:\.\d+)?")

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__module__)

    @staticmethod
    def _require_columns(
        dataframe: pd.DataFrame,
        required_columns: set[str],
        stage_name: str,
    ) -> None:
        """Ensure a cleaning stage received every column it requires."""
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise DataValidationError(
                f"{stage_name} requires missing columns: {missing}"
            )

    def common_drop_and_rename(
        self,
        dataframe: pd.DataFrame,
        rename_mapping: Mapping[str, str],
    ) -> pd.DataFrame:
        """Remove source-only identifiers and standardize shared column names."""
        if not isinstance(dataframe, pd.DataFrame):
            raise DataValidationError("Property input must be a pandas DataFrame")

        cleaned = dataframe.copy()
        cleaned = cleaned.drop(columns=list(self.DROP_COLUMNS), errors="ignore")
        cleaned = cleaned.rename(columns=dict(rename_mapping))

        self.logger.info(
            "Standardized shared property columns; shape=%s",
            cleaned.shape,
        )
        return cleaned

    @classmethod
    def _parse_price(cls, value: object) -> float | None:
        """Convert crore/lac price text into a numeric crore value."""
        if pd.isna(value):
            return None

        text = str(value).strip().lower()
        if not text or "price on request" in text:
            return None

        match = cls.NUMBER_PATTERN.search(text.replace(",", ""))
        if match is None:
            return None

        price = float(match.group())
        if "lac" in text or "lakh" in text:
            price /= 100
        return round(price, 2)

    @classmethod
    def _parse_price_per_sqft(cls, value: object) -> float | None:
        """Extract a numeric price-per-square-foot value from formatted text."""
        if pd.isna(value):
            return None

        text = str(value).replace(",", "")
        match = cls.NUMBER_PATTERN.search(text)
        return round(float(match.group()), 2) if match else None

    @staticmethod
    def _parse_count(value: object) -> float | None:
        """Extract the first integer count from a property attribute."""
        if pd.isna(value):
            return None

        match = re.search(r"\d+", str(value))
        return float(match.group()) if match else None

    @classmethod
    def _parse_balcony(cls, value: object) -> float | None:
        """Convert balcony text, including the 'No Balcony' case, to a count."""
        if pd.isna(value):
            return None

        text = str(value).strip().lower()
        if "no" in text:
            return 0.0
        return cls._parse_count(text)

    def clean_society(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Normalize society names and remove embedded rating text."""
        self._require_columns(dataframe, {"society"}, "Society cleaning")
        cleaned = dataframe.copy()
        cleaned["society"] = (
            cleaned["society"]
            .fillna(self.missing_society_value)
            .astype(str)
            .str.replace(self.RATING_PATTERN, "", regex=True)
            .str.strip()
            .str.lower()
            .replace("", self.missing_society_value)
        )
        return cleaned

    def clean_price(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Normalize the price column to crore values."""
        self._require_columns(dataframe, {"price"}, "Price cleaning")
        cleaned = dataframe.copy()
        cleaned["price"] = cleaned["price"].map(self._parse_price)
        if self.drop_missing_prices:
            cleaned = cleaned.loc[cleaned["price"].notna()].copy()
        return cleaned

    def clean_price_per_sqft(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Normalize price-per-square-foot values."""
        self._require_columns(
            dataframe,
            {"price_per_sqft"},
            "Price-per-square-foot cleaning",
        )
        cleaned = dataframe.copy()
        cleaned["price_per_sqft"] = cleaned["price_per_sqft"].map(
            self._parse_price_per_sqft
        )
        return cleaned

    def clean_bedroom(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Normalize bedroom counts."""
        return self._clean_count_column(dataframe, "bedRoom", "Bedroom cleaning")

    def clean_bathroom(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Normalize bathroom counts."""
        return self._clean_count_column(dataframe, "bathroom", "Bathroom cleaning")

    def _clean_count_column(
        self,
        dataframe: pd.DataFrame,
        column_name: str,
        stage_name: str,
    ) -> pd.DataFrame:
        self._require_columns(dataframe, {column_name}, stage_name)
        cleaned = dataframe.copy()
        cleaned[column_name] = cleaned[column_name].map(self._parse_count)
        return cleaned

    def clean_balcony(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Normalize balcony counts."""
        self._require_columns(dataframe, {"balcony"}, "Balcony cleaning")
        cleaned = dataframe.copy()
        cleaned["balcony"] = cleaned["balcony"].map(self._parse_balcony)
        return cleaned

    def clean_additional_room(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Normalize additional-room descriptions."""
        self._require_columns(
            dataframe,
            {"additionalRoom"},
            "Additional-room cleaning",
        )
        cleaned = dataframe.copy()
        cleaned["additionalRoom"] = (
            cleaned["additionalRoom"]
            .fillna(self.missing_additional_room_value)
            .astype(str)
            .str.strip()
            .str.lower()
        )
        return cleaned

    def clean_facing(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Normalize facing-direction text."""
        self._require_columns(dataframe, {"facing"}, "Facing cleaning")
        cleaned = dataframe.copy()
        cleaned["facing"] = (
            cleaned["facing"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )
        return cleaned

    def insert_derived_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate area and assign the source-specific property type."""
        self._require_columns(
            dataframe,
            {"price", "price_per_sqft"},
            "Derived-feature creation",
        )
        cleaned = dataframe.copy()
        price = pd.to_numeric(cleaned["price"], errors="coerce")
        price_per_sqft = pd.to_numeric(
            cleaned["price_per_sqft"],
            errors="coerce",
        ).replace(0, pd.NA)

        cleaned["area"] = ((price * 10_000_000) / price_per_sqft).round()

        if "property_type" in cleaned.columns:
            cleaned["property_type"] = self.property_type
        else:
            cleaned.insert(1, "property_type", self.property_type)
        return cleaned

    def _run_shared_steps(
        self,
        dataframe: pd.DataFrame,
        rename_mapping: Mapping[str, str],
    ) -> pd.DataFrame:
        cleaned = self.common_drop_and_rename(dataframe, rename_mapping)
        cleaned = self.clean_society(cleaned)
        cleaned = self.clean_price(cleaned)
        cleaned = self.clean_price_per_sqft(cleaned)
        cleaned = self.clean_bedroom(cleaned)
        cleaned = self.clean_bathroom(cleaned)
        cleaned = self.clean_balcony(cleaned)
        cleaned = self.clean_additional_room(cleaned)
        return cleaned


class FlatCleaner(BasePropertyCleaner):
    """Clean the raw flat-listing dataset."""

    property_type = "flat"
    missing_society_value = "unknown"
    missing_additional_room_value = ""
    drop_missing_prices = False

    @staticmethod
    def _parse_floor_info(value: object) -> tuple[float | None, float | None]:
        """Extract current and total floor counts from flat floor text."""
        if pd.isna(value):
            return None, None

        text = str(value).strip().lower()
        current_floor: float | None = None

        if "lower ground" in text:
            current_floor = -0.5
        elif "basement" in text:
            current_floor = -1.0
        elif "ground" in text or text == "g":
            current_floor = 0.0

        parts = re.split(r"\s+of\s+", text, maxsplit=1)
        if current_floor is None:
            current_floor = BasePropertyCleaner._parse_count(parts[0])

        total_floors = (
            BasePropertyCleaner._parse_count(parts[1])
            if len(parts) == 2
            else None
        )
        return current_floor, total_floors

    def clean_floor(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Create current-floor and total-floor columns for flats."""
        self._require_columns(dataframe, {"floorNum"}, "Flat floor cleaning")
        cleaned = dataframe.copy()
        parsed = cleaned["floorNum"].map(self._parse_floor_info)
        cleaned["floor_num"] = parsed.str[0]
        cleaned["total_floors"] = parsed.str[1]
        return cleaned

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Execute the complete flat-cleaning workflow."""
        self.logger.info("Starting flat preprocessing; rows=%d", len(dataframe))
        cleaned = self._run_shared_steps(
            dataframe,
            rename_mapping={"area": "price_per_sqft"},
        )
        cleaned = self.clean_floor(cleaned)
        cleaned = self.clean_facing(cleaned)
        cleaned = self.insert_derived_features(cleaned)
        self.logger.info(
            "Completed flat preprocessing; rows=%d columns=%d",
            *cleaned.shape,
        )
        return cleaned


class HouseCleaner(BasePropertyCleaner):
    """Clean the raw independent-house listing dataset."""

    property_type = "house"
    missing_society_value = "independent"
    missing_additional_room_value = "not available"
    drop_missing_prices = True

    @classmethod
    def _parse_floor(cls, value: object) -> float | None:
        """Extract the house floor count."""
        return cls._parse_count(value)

    def clean_floor(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Rename and normalize the house floor column."""
        self._require_columns(dataframe, {"noOfFloor"}, "House floor cleaning")
        cleaned = dataframe.copy()
        cleaned = cleaned.rename(columns={"noOfFloor": "floorNum"})
        cleaned["floorNum"] = cleaned["floorNum"].map(self._parse_floor)
        return cleaned

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Execute the complete house-cleaning workflow."""
        self.logger.info("Starting house preprocessing; rows=%d", len(dataframe))
        cleaned = self._run_shared_steps(
            dataframe,
            rename_mapping={"rate": "price_per_sqft"},
        )
        cleaned = self.clean_floor(cleaned)
        cleaned = self.clean_facing(cleaned)
        cleaned = self.insert_derived_features(cleaned)
        cleaned["floor_num"] = float("nan")
        cleaned["total_floors"] = float("nan")
        self.logger.info(
            "Completed house preprocessing; rows=%d columns=%d",
            *cleaned.shape,
        )
        return cleaned


class PropertyPreprocessingPipeline:
    """Clean and merge flat and house listings into one dataset."""

    def __init__(
        self,
        flat_cleaner: FlatCleaner | None = None,
        house_cleaner: HouseCleaner | None = None,
    ) -> None:
        self.flat_cleaner = flat_cleaner or FlatCleaner()
        self.house_cleaner = house_cleaner or HouseCleaner()
        self.logger = get_logger(self.__class__.__module__)

    def run(
        self,
        flat_data: pd.DataFrame,
        house_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Clean both source datasets and return a merged property dataset."""
        flat_cleaned = self.flat_cleaner.run(flat_data)
        house_cleaned = self.house_cleaner.run(house_data)

        merged = pd.concat(
            [flat_cleaned, house_cleaned],
            ignore_index=True,
            sort=False,
        )
        self.logger.info(
            "Merged preprocessed property data; rows=%d columns=%d",
            *merged.shape,
        )
        return merged

    @staticmethod
    def save(
        dataframe: pd.DataFrame,
        output_path: Path = Config.PROCESS_MERGE_DATA,
    ) -> Path:
        """Persist the merged dataset to the configured processed-data path."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)
        return path
