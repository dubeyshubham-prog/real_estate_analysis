#-------------------------->
#REQUIRED LIBRARIES
#-------------------------->
import pandas as pd
from config.settings import Config
from src.utils.logger import get_logger

#-------------------------->
#CALLING LOGGER FUNCTION
#-------------------------->
logger = get_logger(__name__)

#-------------------------->
#CENTRALIZED DATA LOADER FOR HOUSING-RELATED CSV FILES
#-------------------------->
class DataLoader:
    # -------------------------->
    #FUNCTION TO LOAD HOUSE DATASET
    # -------------------------->
    def load_house_data(self) -> pd.DataFrame:
        logger.info("Loading raw apartments data...")
        df = pd.read_csv(Config.HOUSE_CSV)
        logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df

    # -------------------------->
    #FUNCTION TO LOAD FLAT DATASET
    # -------------------------->
    def load_flat_data(self) -> pd.DataFrame:
        logger.info("Loading visualization data...")
        df = pd.read_csv(Config.FLAT_CSV)
        logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df

    # -------------------------->
    #FUNCTION TO LOAD CLEANED PROPERTY DATA
    # -------------------------->
    def load_cleaned_property_data(self) -> pd.DataFrame:
        logger.info("Loading visualization data...")
        df = pd.read_csv(Config.PROCESS_MERGE_DATA)
        logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df

    # -------------------------->
    #FUNCTION TO LOAD CLEANED PROPERTY DATA
    # -------------------------->
    def load_gurgaon_property_v1_data(self) -> pd.DataFrame:
        logger.info("Loading visualization data...")
        df = pd.read_csv(Config.GURGAON_PROPERTY_V1)
        logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df

    def load_apartment_data(self) -> pd.DataFrame:
        logger.info("Loading visualization data...")
        df = pd.read_csv(Config.APARTMENTS_CSV)
        logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df

    def load_gurgaon_property_v2_data(self) -> pd.DataFrame:
        logger.info("Loading visualization data...")
        df = pd.read_csv(Config.GURGAON_PROPERTY_V2)
        logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df

    def load_outlier_treated_data(self) -> pd.DataFrame:
        logger.info("Loading visualization data...")
        df = pd.read_csv(Config.OUTLIER_TREATED_CSV)
        logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df

    def load_missing_value_data(self) -> pd.DataFrame:
        logger.info("Loading visualization data...")
        df = pd.read_csv(Config.MISSING_VALUE_TREATED_CSV)
        logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df

    def load_feature_selection_data(self) -> pd.DataFrame:
        logger.info("Loading visualization data...")
        df = pd.read_csv(Config.POST_FEATURE_SELECTION)
        logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df

    def load_gurgaon_property_analysis_data(self) -> pd.DataFrame:
        logger.info("Loading visualization data...")
        df = pd.read_csv(Config.GURGAON_ANALYSIS_PROPERTY)
        logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df

    def load_lat_long_data(self) -> pd.DataFrame:
        logger.info("Loading visualization data...")
        df = pd.read_csv(Config.LAT_LONG_DATA)
        logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df

