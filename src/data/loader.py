#--------------------------->
#ALL THE REQUIRED LIBRARIES:
#--------------------------->
from pathlib import Path
from typing import Any
import pandas as pd
from config.settings import Config
from src.common.exceptions import DataLoadError
from src.monitoring.logging import get_logger

#--------------------------->
#DATA LOADER CLASS:
#--------------------------->
class DataLoader:
    """Load project datasets through one validated CSV-reading interface."""
    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def _load_csv(self,file_path: Path,dataset_name: str, **read_csv_options: Any,) -> pd.DataFrame:
        """Load a CSV file and raise a domain-specific error when it fails."""
        path = Path(file_path)

        if not path.is_file():
            raise DataLoadError(
                f"{dataset_name} dataset was not found at: {path}"
            )

        self.logger.info(
            "Loading %s dataset from %s",
            dataset_name,
            path,
        )

        try:
            dataframe = pd.read_csv(path, **read_csv_options)
        except (
            OSError,
            UnicodeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as error:
            self.logger.exception(
                "Failed to load %s dataset from %s",
                dataset_name,
                path,
            )
            raise DataLoadError(
                f"Unable to load {dataset_name} dataset from: {path}"
            ) from error

        if dataframe.empty:
            raise DataLoadError(
                f"{dataset_name} dataset is empty: {path}"
            )

        self.logger.info(
            "Loaded %s dataset with %d rows and %d columns",
            dataset_name,
            dataframe.shape[0],
            dataframe.shape[1],
        )
        return dataframe

    def load_house_data(self) -> pd.DataFrame:
        return self._load_csv(Config.HOUSE_CSV, "raw house")

    def load_flat_data(self) -> pd.DataFrame:
        return self._load_csv(Config.FLAT_CSV, "raw flat")

    def load_apartment_data(self) -> pd.DataFrame:
        return self._load_csv(Config.APARTMENTS_CSV, "apartment metadata")

    def load_cleaned_property_data(self) -> pd.DataFrame:
        return self._load_csv(Config.PROCESS_MERGE_DATA, "merged property")

    def load_gurgaon_property_v1_data(self) -> pd.DataFrame:
        return self._load_csv(
            Config.GURGAON_PROPERTY_V1,
            "level-one cleaned property",
        )

    def load_gurgaon_property_v2_data(self) -> pd.DataFrame:
        return self._load_csv(
            Config.GURGAON_PROPERTY_V2,
            "feature-engineered property",
        )

    def load_outlier_treated_data(self) -> pd.DataFrame:
        return self._load_csv(
            Config.OUTLIER_TREATED_CSV,
            "outlier-treated property",
        )

    def load_missing_value_data(self) -> pd.DataFrame:
        return self._load_csv(
            Config.MISSING_VALUE_TREATED_CSV,
            "missing-value-treated property",
        )

    def load_feature_selection_data(self) -> pd.DataFrame:
        return self._load_csv(
            Config.POST_FEATURE_SELECTION,
            "feature-selected property",
        )

    def load_gurgaon_property_analysis_data(self) -> pd.DataFrame:
        return self._load_csv(
            Config.GURGAON_ANALYSIS_PROPERTY,
            "Gurgaon market analysis",
        )

    def load_lat_long_data(self) -> pd.DataFrame:
        return self._load_csv(Config.LAT_LONG_DATA, "sector coordinates")

