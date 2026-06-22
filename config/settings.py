#----------------------------->
#REQUIRED LIBRARIES ARE HERE:
#----------------------------->
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

#----------------------------->
#ROOT ADDRESS:
#----------------------------->
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME_DIR = Path(
    os.getenv("RUNTIME_DIR", str(PROJECT_ROOT / "data"))
).resolve()

#----------------------------->
#LOADING ENVIRONMENTAL VARIABLE:
#----------------------------->
load_dotenv(PROJECT_ROOT / ".env")

#----------------------------->
#CONVERT A STRING INTO BOOLEAN:
#----------------------------->
def _get_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable using common truthy values."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

#----------------------------->
'''
CONVERT A LOG LEVEL TEXT INTO
A NUMERIC VALUE WHICH MAKE
SENSE TO THE PYTHON INTERPRETER
'''
#----------------------------->
def _get_log_level(name: str, default: str = "INFO") -> int:
    """Convert a configured log-level name into a logging constant."""
    level_name = os.getenv(name, default).strip().upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level configured in {name}: {level_name}")
    return level


class Config:
    """Central application configuration loaded from environment variables."""

    # Application
    APP_NAME: str = os.getenv(
        "APP_NAME",
        "PropLens",
    )
    APP_TAGLINE: str = os.getenv(
        "APP_TAGLINE",
        "Real Estate Intelligence, Clearly.",
    )
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = _get_bool("DEBUG", default=False)
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    ENABLE_VISION: bool = _get_bool("ENABLE_VISION", default=True)
    USE_OLLAMA_ROUTER: bool = _get_bool(
        "USE_OLLAMA_ROUTER",
        default=False,
    )
    RUNTIME_DIR: Path = DEFAULT_RUNTIME_DIR
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{(DEFAULT_RUNTIME_DIR / 'proplens.db').as_posix()}",
    )

    # Project directories
    BASE_DIR: Path = PROJECT_ROOT
    APP_DIR: Path = BASE_DIR / "app"
    SRC_DIR: Path = BASE_DIR / "src"
    CONFIG_DIR: Path = BASE_DIR / "config"
    DATA_DIR: Path = BASE_DIR / "data"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    LOGS_DIR: Path = RUNTIME_DIR / "logs"
    REPORTS_DIR: Path = BASE_DIR / "reports"

    # Application resources
    TEMPLATES_DIR: Path = APP_DIR / "templates"
    STATIC_DIR: Path = APP_DIR / "static"
    UPLOADS_DIR: Path = RUNTIME_DIR / "uploads"
    IMAGE_UPLOADS_DIR: Path = UPLOADS_DIR / "images"
    DOCUMENT_UPLOADS_DIR: Path = UPLOADS_DIR / "documents"
    PROPERTY_IMAGES_RAW_DIR: Path = DATA_DIR / "property_images" / "raw"
    ROOM_DATASET_DIR: Path = PROPERTY_IMAGES_RAW_DIR / "room-dataset"
    PROPERTY_IMAGES_DIR: Path = ROOM_DATASET_DIR
    CHROMA_DB_DIR: Path = RUNTIME_DIR / "chroma_db"
    RAG_COLLECTION_NAME: str = os.getenv(
        "RAG_COLLECTION_NAME",
        "real_estate_knowledge",
    )
    RAG_RESULT_LIMIT: int = int(os.getenv("RAG_RESULT_LIMIT", "3"))
    WEB_SEARCH_URL: str = os.getenv(
        "WEB_SEARCH_URL",
        "https://html.duckduckgo.com/html/",
    )
    WEB_SEARCH_FALLBACK_URL: str = os.getenv(
        "WEB_SEARCH_FALLBACK_URL",
        "https://www.bing.com/news/search",
    )
    WEB_SEARCH_TIMEOUT_SECONDS: int = int(
        os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "10")
    )

    # Machine-learning artifacts
    PRICE_MODEL_DIR: Path = ARTIFACTS_DIR / "models" / "price_prediction"
    PRICE_MODEL_FILE: Path = PRICE_MODEL_DIR / "model.joblib"
    PRICE_REFERENCE_DATA_FILE: Path = PRICE_MODEL_DIR / "reference_data.joblib"
    PRICE_MODEL_METADATA_FILE: Path = PRICE_MODEL_DIR / "metadata.json"
    MODEL_EVALUATION_REPORT: Path = REPORTS_DIR / "model_evaluation.csv"
    HYPERPARAMETER_REPORT: Path = REPORTS_DIR / "hyperparameter_tuning.csv"

    # Recommendation artifacts
    RECOMMENDATION_MODEL_DIR: Path = (
        ARTIFACTS_DIR / "models" / "recommendation"
    )
    RECOMMENDATION_MODEL_FILE: Path = (
        RECOMMENDATION_MODEL_DIR / "hybrid_recommender.joblib"
    )
    RECOMMENDATION_METADATA_FILE: Path = (
        RECOMMENDATION_MODEL_DIR / "metadata.json"
    )

    # Deep-learning artifacts
    VISION_MODEL_DIR: Path = ARTIFACTS_DIR / "models" / "vision"
    ROOM_CLASSIFIER_MODEL_FILE: Path = (
        VISION_MODEL_DIR / "room_classifier.pth"
    )
    VISUAL_EMBEDDING_DATABASE_FILE: Path = (
        VISION_MODEL_DIR / "visual_embedding_database.pkl"
    )

    # Data paths
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    HOUSE_CSV: Path = RAW_DATA_DIR / "house.csv"
    FLAT_CSV: Path = RAW_DATA_DIR / "flat.csv"
    APARTMENTS_CSV: Path = RAW_DATA_DIR / "apartments.csv"
    PROCESS_MERGE_DATA: Path = PROCESSED_DATA_DIR / "cleaned_properties.csv"
    GURGAON_PROPERTY_V1: Path = (
        PROCESSED_DATA_DIR / "gurgaon_properties_cleaned_v1.csv"
    )
    GURGAON_PROPERTY_V2: Path = (
        PROCESSED_DATA_DIR / "gurgaon_properties_cleaned_v2.csv"
    )
    OUTLIER_TREATED_CSV: Path = (
        PROCESSED_DATA_DIR / "gurgaon_properties_outlier_treated.csv"
    )
    MISSING_VALUE_TREATED_CSV: Path = (
        PROCESSED_DATA_DIR / "gurgaon_properties_missing_value_imputation.csv"
    )
    POST_FEATURE_SELECTION: Path = (
        PROCESSED_DATA_DIR / "gurgaon_properties_post_feature_selection.csv"
    )
    FEATURE_IMPORTANCE_REPORT: Path = (
        REPORTS_DIR / "feature_importance.csv"
    )
    GURGAON_ANALYSIS_PROPERTY: Path = RAW_DATA_DIR / "gurgaon_property.csv"
    LAT_LONG_DATA: Path = RAW_DATA_DIR / "latlongs.csv"

    # Logging
    LOG_LEVEL: int = _get_log_level("LOG_LEVEL")
    LOG_FORMAT: str = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_FILE: Path = LOGS_DIR / "app.log"

    @classmethod
    def initialize_directories(cls) -> None:
        """Create directories that the running application may write to."""
        runtime_directories = [
            cls.ARTIFACTS_DIR,
            cls.LOGS_DIR,
            cls.REPORTS_DIR,
            cls.UPLOADS_DIR,
            cls.IMAGE_UPLOADS_DIR,
            cls.DOCUMENT_UPLOADS_DIR,
            cls.CHROMA_DB_DIR,
            cls.VISION_MODEL_DIR,
        ]
        for directory in runtime_directories:
            directory.mkdir(parents=True, exist_ok=True)
