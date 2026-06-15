from pathlib import Path
import logging

class Config:
    """Base configuration class holding project constants and paths."""
    # Base Directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_DIR: Path = BASE_DIR / 'models'
    DATA_DIR: Path = BASE_DIR / 'data'
    SRC_DIR: Path = BASE_DIR / 'src'
    LOGS_DIR: Path = BASE_DIR / "logs"

    # Data Paths
    HOUSE_CSV: Path = DATA_DIR / 'raw' / 'house.csv'
    FLAT_CSV: Path = DATA_DIR / 'raw' / 'flat.csv'
    APARTMENTS_CSV: Path = DATA_DIR / 'raw' / 'apartments.csv'
    PROCESS_MERGE_DATA: Path = DATA_DIR / 'processed' / 'cleaned_properties.csv'
    GURGAON_PROPERTY_V1: Path = DATA_DIR / 'processed' / 'gurgaon_properties_cleaned_v1.csv'
    GURGAON_PROPERTY_V2: Path = DATA_DIR / 'processed' / 'gurgaon_properties_cleaned_v2.csv'
    OUTLIER_TREATED_CSV: Path = DATA_DIR / 'processed' / 'gurgaon_properties_outlier_treated.csv'
    MISSING_VALUE_TREATED_CSV: Path = DATA_DIR / 'processed' / 'gurgaon_properties_missing_value_imputation.csv'
    POST_FEATURE_SELECTION: Path = DATA_DIR / 'processed' / 'gurgaon_properties_post_feature_selection.csv'
    GURGAON_ANALYSIS_PROPERTY: Path = DATA_DIR / 'raw' / 'gurgaon_property.csv'
    LAT_LONG_DATA: Path = DATA_DIR / 'raw' / 'latlongs.csv'

    # Logging Parameters
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(filename)s | %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_FILE: Path = LOGS_DIR / "app.log"

    @classmethod
    def initialize_directories(cls) -> None:
        """Ensures required project directories exist before code runs."""
        for directory in [cls.MODEL_DIR, cls.DATA_DIR, cls.SRC_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
