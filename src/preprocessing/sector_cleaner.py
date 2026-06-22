from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import pandas as pd

from config.settings import Config
from src.common.exceptions import DataValidationError
from src.monitoring.logging import get_logger


class SectorPreprocessor:
    """Extract, standardize, and filter Gurgaon property sectors."""

    DEFAULT_MINIMUM_LISTINGS: ClassVar[int] = 3
    DROP_COLUMNS: ClassVar[tuple[str, ...]] = (
        "property_name",
        "description",
        "rating",
    )

    SECTOR_MAPPING: ClassVar[dict[str, str]] = {
        "dharam colony": "sector 12",
        "krishna colony": "sector 7",
        "suncity": "sector 54",
        "prem nagar": "sector 13",
        "mg road": "sector 28",
        "gandhi nagar": "sector 28",
        "laxmi garden": "sector 11",
        "shakti nagar": "sector 11",
        "baldev nagar": "sector 7",
        "shivpuri": "sector 7",
        "garhi harsaru": "sector 17",
        "imt manesar": "manesar",
        "adarsh nagar": "sector 12",
        "shivaji nagar": "sector 11",
        "bhim nagar": "sector 6",
        "madanpuri": "sector 7",
        "saraswati vihar": "sector 28",
        "arjun nagar": "sector 8",
        "ravi nagar": "sector 9",
        "vishnu garden": "sector 105",
        "bhondsi": "sector 11",
        "surya vihar": "sector 21",
        "devilal colony": "sector 9",
        "valley view estate": "gwal pahari",
        "mehrauli  road": "sector 14",
        "jyoti park": "sector 7",
        "ansal plaza": "sector 23",
        "dayanand colony": "sector 6",
        "sushant lok phase 2": "sector 55",
        "chakkarpur": "sector 28",
        "greenwood city": "sector 45",
        "subhash nagar": "sector 12",
        "sohna road road": "sohna road",
        "malibu town": "sector 47",
        "surat nagar 1": "sector 104",
        "new colony": "sector 7",
        "mianwali colony": "sector 12",
        "jacobpura": "sector 12",
        "rajiv nagar": "sector 13",
        "ashok vihar": "sector 3",
        "dlf phase 1": "sector 26",
        "nirvana country": "sector 50",
        "palam vihar": "sector 2",
        "dlf phase 2": "sector 25",
        "sushant lok phase 1": "sector 43",
        "laxman vihar": "sector 4",
        "dlf phase 4": "sector 28",
        "dlf phase 3": "sector 24",
        "sushant lok phase 3": "sector 57",
        "dlf phase 5": "sector 43",
        "rajendra park": "sector 105",
        "uppals southend": "sector 49",
        "sohna": "sohna road",
        "ashok vihar phase 3 extension": "sector 5",
        "south city 1": "sector 41",
        "ashok vihar phase 2": "sector 5",
        "sector 95a": "sector 95",
        "sector 23a": "sector 23",
        "sector 12a": "sector 12",
        "sector 3a": "sector 3",
        "sector 110 a": "sector 110",
        "patel nagar": "sector 15",
        "a block sector 43": "sector 43",
        "maruti kunj": "sector 12",
        "b block sector 43": "sector 43",
        "sector-33 sohna road": "sector 33",
        "sector 1 manesar": "manesar",
        "sector 4 phase 2": "sector 4",
        "sector 1a manesar": "manesar",
        "c block sector 43": "sector 43",
        "sector 89 a": "sector 89",
        "sector 2 extension": "sector 2",
        "sector 36 sohna road": "sector 36",
    }

    # These corrections depend on the current source row index. They are kept
    # isolated until a stable listing identifier is introduced upstream.
    INDEX_SECTOR_FIXES: ClassVar[dict[int, str]] = {
        955: "sector 37",
        2800: "sector 92",
        2838: "sector 90",
        2857: "sector 76",
        311: "sector 110",
        1072: "sector 110",
        1486: "sector 110",
        3040: "sector 110",
        3875: "sector 110",
    }

    def __init__(
        self,
        minimum_listings: int = DEFAULT_MINIMUM_LISTINGS,
        sector_mapping: Mapping[str, str] | None = None,
    ) -> None:
        if minimum_listings < 1:
            raise ValueError("minimum_listings must be at least 1")

        self.minimum_listings = minimum_listings
        self.sector_mapping = dict(sector_mapping or self.SECTOR_MAPPING)
        self.logger = get_logger(__name__)

    @staticmethod
    def _require_columns(
        dataframe: pd.DataFrame,
        required_columns: set[str],
        stage_name: str,
    ) -> None:
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise DataValidationError(
                f"{stage_name} requires missing columns: {missing}"
            )

    def create_sector_column(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract the location appearing after 'in' in each property name."""
        self._require_columns(
            dataframe,
            {"property_name"},
            "Sector extraction",
        )
        cleaned = dataframe.copy()

        if "sector" in cleaned.columns:
            raise DataValidationError(
                "Sector extraction cannot insert an existing 'sector' column"
            )

        extracted_sector = (
            cleaned["property_name"]
            .astype("string")
            .str.extract(r"\bin\s+(.+)$", expand=False)
            .str.replace("Gurgaon", "", case=False, regex=False)
            .str.strip()
            .str.lower()
        )

        if extracted_sector.isna().any():
            invalid_rows = int(extracted_sector.isna().sum())
            raise DataValidationError(
                f"Sector extraction failed for {invalid_rows} property names"
            )

        cleaned.insert(min(3, len(cleaned.columns)), "sector", extracted_sector)
        return cleaned

    def clean_sector(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Map neighborhood aliases to standardized Gurgaon sectors."""
        self._require_columns(dataframe, {"sector"}, "Sector standardization")
        cleaned = dataframe.copy()
        cleaned["sector"] = cleaned["sector"].replace(self.sector_mapping)
        return cleaned

    def remove_rare_sectors(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Keep sectors meeting the configured minimum listing count."""
        self._require_columns(dataframe, {"sector"}, "Rare-sector filtering")
        sector_counts = dataframe["sector"].value_counts()
        valid_sectors = sector_counts[
            sector_counts >= self.minimum_listings
        ].index
        return dataframe.loc[dataframe["sector"].isin(valid_sectors)].copy()

    def apply_manual_fixes(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Apply known source-row corrections that survive prior filtering."""
        self._require_columns(dataframe, {"sector"}, "Manual sector correction")
        cleaned = dataframe.copy()

        applicable_fixes = {
            index: sector
            for index, sector in self.INDEX_SECTOR_FIXES.items()
            if index in cleaned.index
        }
        skipped_count = len(self.INDEX_SECTOR_FIXES) - len(applicable_fixes)

        if applicable_fixes:
            fix_indexes = list(applicable_fixes)
            cleaned.loc[fix_indexes, "sector"] = [
                applicable_fixes[index] for index in fix_indexes
            ]

        if skipped_count:
            self.logger.warning(
                "Skipped %d index-based sector corrections because their rows "
                "were unavailable after filtering",
                skipped_count,
            )

        return cleaned

    def drop_unused_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Remove text and source columns not needed by later ML stages."""
        self._require_columns(
            dataframe,
            set(self.DROP_COLUMNS),
            "Unused-column removal",
        )
        return dataframe.drop(columns=list(self.DROP_COLUMNS)).copy()

    # Kept as a compatibility wrapper for existing callers.
    def drop_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return self.drop_unused_columns(dataframe)

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Execute the complete second-level preprocessing workflow."""
        self.logger.info(
            "Starting sector preprocessing; rows=%d",
            len(dataframe),
        )
        cleaned = self.create_sector_column(dataframe)
        cleaned = self.clean_sector(cleaned)
        cleaned = self.remove_rare_sectors(cleaned)
        cleaned = self.apply_manual_fixes(cleaned)
        cleaned = self.drop_unused_columns(cleaned)
        self.logger.info(
            "Completed sector preprocessing; rows=%d columns=%d sectors=%d",
            cleaned.shape[0],
            cleaned.shape[1],
            cleaned["sector"].nunique(),
        )
        return cleaned

    # Kept as a compatibility wrapper for the former API.
    def run_pipeline(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return self.run(dataframe)

    @staticmethod
    def save(
        dataframe: pd.DataFrame,
        output_path: Path = Config.GURGAON_PROPERTY_V1,
    ) -> Path:
        """Save the standardized sector dataset."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)
        return path


# Temporary compatibility alias; new code should use SectorPreprocessor.
PreprocessorLevel2 = SectorPreprocessor
