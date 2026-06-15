#-------------------------->
#REQUIRED LIBRARIES
#-------------------------->
import pandas as pd
import re
from streamlit import columns
from config.settings import Config
from src.utils.logger import get_logger

#-------------------------->
#INITIALIZING LOGGER TO MONITOR EVERY STEP
#-------------------------->
logger = get_logger(__name__)

#-------------------------->
'''
WRITE A COMMON CLASS WHICH CAN
HANDLE COMMON CLEANING STEP  IN 
BOTH THE DATASET
'''
#-------------------------->
class BasePropertyCleaner:
    def __init__(self) -> None:
        pass

    def common_drop_and_rename(
        self,
        df: pd.DataFrame,
        rename_mapping: dict
    ) -> pd.DataFrame:

        logger.info(
            "BaseClass Execution: To drop and rename shared column in both the datasets"
        )

        df = df.copy()

        df.drop(
            columns=['link', 'property_id'],
            errors='ignore',
            inplace=True
        )

        df.rename(
            columns=rename_mapping,
            inplace=True
        )

        return df
#-------------------------->
'''
WRITE A CHILD CLASS WHICH 
HANDLES THE DATA PREPROCESSING
PROCESS OF THE FLAT DATASET
'''
#-------------------------->
class FlatCleaner(BasePropertyCleaner):
    def __init__(self):
        #SUPER KEYWORD IS USED TO ACTIVATE THE FEATURES OF THE PARENT CLASS
        super().__init__()

    def clean_society(self,df:pd.DataFrame)->pd.DataFrame:
        logger.info("FlatCleaner Execution: Cleaning the society column...")

        # MAKE A COPY TO PROTECT ORIGINAL DATASET:
        df = df.copy()

        df['society'] = df['society'].apply(
            lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()
        ).str.lower()

        return df

    def _parse_price(self, price_str: str) -> float:
        """CONVERTS PRICE STRINGS LIKE '1.5 CRORE' OR '90 LAC' TO A FLOAT VALUE IN CRORES."""
        # CHECK FOR MISSING OR INVALID DATA TYPES
        if pd.isna(price_str) or not isinstance(price_str, str):
            return None

        price_str = price_str.lower().strip()
        # HANDLE ANOMALOUS PRICE ON REQUEST RECORDS
        if 'price on request' in price_str:
            return None

        try:
            # EXTRACT THE NUMERIC PART USING STRING COMPREHENSION
            value = float(''.join(c for c in price_str if c.isdigit() or c == '.'))

            # CONVERT LAC TO CRORE UNIT IF APPLICABLE
            if 'lac' in price_str:
                return round(value / 100, 2)
            # RETURN DEFAULT CRORE VALUE
            return round(value, 2)
        except ValueError:
            # HANDLE UNEXPECTED PARSING ERRORS SAFELY
            return None

    def clean_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """APPLIES THE PRICE PARSER TO THE DATAFRAME."""
        # CREATE A DEEP MEMORY COPY TO PREVENT SETTINGWITHCOPYWARNING
        df = df.copy()
        # APPLY THE PARSER ROW-WISE ON THE PRICE COLUMN
        df['price'] = df['price'].apply(self._parse_price)
        return df

    # ==========================================================
    # PRICE PER SQFT COLUMN CLEANING
    # ==========================================================
    def _parse_price_per_sqft(self, value: str) -> float:

        if pd.isna(value):
            return None

        try:
            value = str(value)

            value = (
                value
                .replace("₹", "")
                .replace("/sq.ft.", "")
                .replace(",", "")
                .strip()
            )

            return float(value)

        except:
            return None

    def clean_price_per_sqft(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info(
            "HouseCleaner Execution: Cleaning the price_per_sqft column..."
        )

        df = df.copy()

        print("\n===== BEFORE CLEANING =====")
        print(df["price_per_sqft"].head())
        print(df["price_per_sqft"].dtype)

        df["price_per_sqft"] = (
            df["price_per_sqft"]
            .apply(self._parse_price_per_sqft)
        )

        print("\n===== AFTER CLEANING =====")
        print(df["price_per_sqft"].head())
        print(df["price_per_sqft"].dtype)

        return df

    def _parse_bedroom(self, room_str: str) -> float:
        """CONVERTS ROOM STRINGS TO CLEAN NUMERIC FLOATS."""
        if pd.isna(room_str) or not isinstance(room_str, str):
            return None

        room_str = room_str.lower().strip()
        try:
            # EXTRACT ONLY DIGITS
            value = ''.join(c for c in room_str if c.isdigit())
            return float(value) if value else None
        except ValueError:
            return None

    def clean_bedroom(self, df: pd.DataFrame) -> pd.DataFrame:
        """APPLIES THE BEDROOM PARSER TO THE DATAFRAME."""
        df = df.copy()
        df['bedRoom'] = df['bedRoom'].apply(self._parse_bedroom)
        return df

    def _parse_bathroom(self, bath_str: str) -> float:
        """CONVERTS BATHROOM STRINGS TO CLEAN NUMERIC FLOATS."""
        if pd.isna(bath_str) or not isinstance(bath_str, str):
            return None

        bath_str = bath_str.lower().strip()
        try:
            # EXTRACT ONLY DIGITS
            value = ''.join(c for c in bath_str if c.isdigit())
            return float(value) if value else None
        except ValueError:
            return None

    def clean_bathroom(self, df: pd.DataFrame) -> pd.DataFrame:
        """APPLIES THE BATHROOM PARSER TO THE DATAFRAME."""
        df = df.copy()
        df['bathroom'] = df['bathroom'].apply(self._parse_bathroom)
        return df

    def _parse_balcony(self, balcony_str: str) -> float:
        """CONVERTS BALCONY STRINGS TO CLEAN NUMERIC FLOATS."""
        if pd.isna(balcony_str) or not isinstance(balcony_str, str):
            return None

        balcony_str = balcony_str.lower().strip()

        # HANDLE 'NO' BALCONY CASE SPECIFICALLY
        if 'no' in balcony_str:
            return 0.0

        try:
            # EXTRACT ONLY DIGITS
            value = ''.join(c for c in balcony_str if c.isdigit())
            return float(value) if value else None
        except ValueError:
            return None

    def clean_balcony(self, df: pd.DataFrame) -> pd.DataFrame:
        """APPLIES THE BALCONY PARSER TO THE DATAFRAME."""
        df = df.copy()
        df['balcony'] = df['balcony'].apply(self._parse_balcony)
        return df

    def clean_additional_room(self, df: pd.DataFrame) -> pd.DataFrame:
        """STANDARDS THE ADDITIONAL ROOM STRINGS BY CLEANING TEXT WHITESPACES."""
        df = df.copy()
        # FILL MISSING VALUES WITH EMPTY STRING AND STRIP TEXT
        df['additionalRoom'] = df['additionalRoom'].fillna('').astype(str).str.lower().str.strip()
        return df

    def _parse_floor_info(self, floor_str: str) -> tuple:
        """EXTRACTS BOTH FLOOR NUMBER AND TOTAL FLOORS FROM A STRING."""
        if pd.isna(floor_str) or not isinstance(floor_str, str):
            return None, None

        floor_str = floor_str.lower().strip()

        # MAP TEXT-BASED FLOORS TO NUMERIC REPRESENTATIONS
        floor_num = None
        if 'ground' in floor_str or 'g' == floor_str:
            floor_num = 0.0
        elif 'basement' in floor_str:
            floor_num = -1.0
        elif 'lower ground' in floor_str:
            floor_num = -0.5

        try:
            # SPLIT BY 'OF' TO SEPARATE CURRENT FLOOR FROM TOTAL FLOORS
            parts = floor_str.split('of')

            # IF FLOOR NUM WAS NOT MAPPED BY TEXT, EXTRACT DIGITS FROM FIRST PART
            if floor_num is None and len(parts) > 0:
                digits = ''.join(c for c in parts[0] if c.isdigit())
                if digits:
                    floor_num = float(digits)

            # EXTRACT TOTAL FLOORS FROM THE SECOND PART
            total_floors = None
            if len(parts) > 1:
                total_digits = ''.join(c for c in parts[1] if c.isdigit())
                if total_digits:
                    total_floors = float(total_digits)

            return floor_num, total_floors
        except Exception:
            return None, None

    def clean_floor(self, df: pd.DataFrame) -> pd.DataFrame:
        """APPLIES THE FLOOR PARSER AND CREATES TWO NEW SEPARATE COLUMNS."""
        df = df.copy()
        # APPLY PARSER TO EXTRACT TUPLE OF (FLOOR_NUM, TOTAL_FLOORS)
        parsed_floors = df['floorNum'].apply(self._parse_floor_info)

        # ASSIGN EXTRACTED VALUES TO INDEPENDENT COLUMNS
        df['floor_num'] = parsed_floors.apply(lambda x: x[0])
        df['total_floors'] = parsed_floors.apply(lambda x: x[1])
        return df

    def clean_facing(self, df: pd.DataFrame) -> pd.DataFrame:
        """STANDARDIZES FACING DIRECTION STRINGS AND HANDLES MISSING VALUES."""
        df = df.copy()
        # FILL MISSING VALUES WITH UNKNOWN AND STRIP TEXT WHITESPACES
        df['facing'] = df['facing'].fillna('Unknown').astype(str).str.strip()
        return df

    def insert_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """COMPUTES AREA MATHEMATICALLY AND INSERTS DERIVED COLUMNS AT EXACT INDEXES."""
        df = df.copy()

        # COERCE COLUMNS TO NUMERIC TO PREVENT MATHEMATICAL OPERATIONS CRASHING
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['price_per_sqft'] = pd.to_numeric(df['price_per_sqft'], errors='coerce')

        # CALCULATE AREA AND INSERT AT LOCATION 4
        calculated_area = round((df['price'] * 10000000) / df['price_per_sqft'])
        df.insert(loc=4, column='area', value=calculated_area)

        # INSERT CONSTANT PROPERTY TYPE AT LOCATION 1
        df.insert(loc=1, column='property_type', value='flat')

        return df

class HouseCleaner(BasePropertyCleaner):

    def __init__(self):
        super().__init__()

    def clean_society(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CLEANS SOCIETY NAMES BY REMOVING RATINGS,
        CONVERTING TO LOWERCASE AND HANDLING MISSING VALUES.
        """

        logger.info("HouseCleaner Execution: Cleaning the society column...")
        df = df.copy()
        df['society'] = df['society'].apply(
            lambda name: re.sub(
                r'\d+(\.\d+)?\s?★',
                '',
                str(name)
            ).strip()
        ).str.lower()

        df['society'] = df['society'].replace(
            'nan',
            'independent'
        )

        return df

    def _parse_price(self, price_str: str) -> float:
        """
        CONVERTS HOUSE PRICE STRINGS TO CRORE FORMAT.
        """
        if pd.isna(price_str) or not isinstance(price_str, str):
            return None

        price_str = price_str.lower().strip()

        if 'price on request' in price_str:
            return None

        try:
            value = float(
                ''.join(
                    c for c in price_str
                    if c.isdigit() or c == '.'
                )
            )

            if 'lac' in price_str:
                return round(value / 100, 2)

            return round(value, 2)

        except ValueError:
            return None

    def clean_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CLEANS PRICE COLUMN AND REMOVES
        PRICE ON REQUEST RECORDS.
        """

        logger.info("HouseCleaner Execution: Cleaning the price column...")

        df = df.copy()

        df['price'] = df['price'].apply(self._parse_price)

        df = df[df['price'].notnull()]

        return df

    def _parse_price_per_sqft(self, price_sqft_str: str) -> float:
        """
        CONVERTS PRICE PER SQFT STRINGS TO A CLEAN FLOAT VALUE.
        HANDLES FORMAT LIKE '₹ 20,115/sq.ft.' BY SPLITTING ON '/'
        AND STRIPPING CURRENCY SYMBOLS AND COMMAS BEFORE PARSING.
        """

        # HANDLE NULL OR INVALID TYPES
        if pd.isna(price_sqft_str) or not isinstance(price_sqft_str, str):
            return None

        try:
            # SPLIT ON '/' TO ISOLATE THE NUMERIC PART BEFORE '/sq.ft.'
            numeric_part = price_sqft_str.split('/')[0]

            # STRIP CURRENCY SYMBOL, COMMAS AND WHITESPACE
            numeric_part = (
                numeric_part
                .replace('₹', '')
                .replace(',', '')
                .strip()
            )

            return round(float(numeric_part), 2)

        except ValueError:
            return None

    def clean_price_per_sqft(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        APPLIES THE PRICE PER SQFT PARSER TO THE DATAFRAME.
        """

        logger.info(
            "HouseCleaner Execution: Cleaning the price_per_sqft column..."
        )

        df = df.copy()

        df['price_per_sqft'] = df['price_per_sqft'].apply(
            self._parse_price_per_sqft
        )

        return df

    def _parse_bedroom(self, room_str: str) -> float:
        """
        CONVERTS BEDROOM STRINGS TO CLEAN NUMERIC VALUES.
        EXAMPLES:
        '3 Bedroom' -> 3
        '5 BHK' -> 5
        """

        # HANDLE NULL OR INVALID TYPES
        if pd.isna(room_str) or not isinstance(room_str, str):
            return None

        room_str = room_str.lower().strip()

        try:
            # EXTRACT ONLY DIGITS
            value = ''.join(
                c for c in room_str
                if c.isdigit()
            )

            return float(value) if value else None

        except ValueError:
            return None

    def clean_bedroom(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        APPLIES BEDROOM PARSER ON ENTIRE COLUMN.
        """

        logger.info(
            "HouseCleaner Execution: Cleaning the bedroom column..."
        )

        # CREATE COPY TO PROTECT ORIGINAL DATAFRAME
        df = df.copy()

        # APPLY PARSER
        df['bedRoom'] = df['bedRoom'].apply(
            self._parse_bedroom
        )

        return df

    def _parse_bathroom(self, bath_str: str) -> float:
        """
        CONVERTS BATHROOM STRINGS TO CLEAN NUMERIC VALUES.
        """

        if pd.isna(bath_str) or not isinstance(bath_str, str):
            return None

        bath_str = bath_str.lower().strip()

        try:
            value = ''.join(
                c for c in bath_str
                if c.isdigit()
            )

            return float(value) if value else None

        except ValueError:
            return None

    def clean_bathroom(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        APPLIES BATHROOM PARSER ON ENTIRE COLUMN.
        """

        logger.info(
            "HouseCleaner Execution: Cleaning the bathroom column..."
        )

        df = df.copy()

        df['bathroom'] = df['bathroom'].apply(
            self._parse_bathroom
        )

        return df

    # ==========================================================
    # BALCONY COLUMN CLEANING
    # ==========================================================

    def _parse_balcony(self, balcony_str: str) -> float:

        if pd.isna(balcony_str) or not isinstance(balcony_str, str):
            return None

        balcony_str = balcony_str.lower().strip()

        if 'no' in balcony_str:
            return 0.0

        try:
            value = ''.join(
                c for c in balcony_str
                if c.isdigit()
            )

            return float(value) if value else None

        except ValueError:
            return None

    def clean_balcony(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info(
            "HouseCleaner Execution: Cleaning the balcony column..."
        )

        df = df.copy()

        df['balcony'] = df['balcony'].apply(
            self._parse_balcony
        )

        return df

    # ==========================================================
    # ADDITIONAL ROOM COLUMN CLEANING
    # ==========================================================

    def clean_additional_room(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info(
            "HouseCleaner Execution: Cleaning the additional room column..."
        )

        df = df.copy()

        df['additionalRoom'] = (
            df['additionalRoom']
            .fillna('not available')
            .astype(str)
            .str.lower()
            .str.strip()
        )

        return df

    # ==========================================================
    # FLOOR COLUMN CLEANING
    # ==========================================================

    def _parse_floor(
            self,
            floor_str: str
    ) -> float:

        if pd.isna(floor_str) or not isinstance(floor_str, str):
            return None

        floor_str = floor_str.lower().strip()

        try:
            value = ''.join(
                c for c in floor_str
                if c.isdigit()
            )

            return float(value) if value else None

        except ValueError:
            return None

    def clean_floor(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info(
            "HouseCleaner Execution: Cleaning the floor column..."
        )

        df = df.copy()

        df.rename(
            columns={
                'noOfFloor': 'floorNum'
            },
            inplace=True
        )

        df['floorNum'] = df['floorNum'].apply(
            self._parse_floor
        )

        return df

    # ==========================================================
    # FACING COLUMN CLEANING
    # ==========================================================

    def clean_facing(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info(
            "HouseCleaner Execution: Cleaning the facing column..."
        )

        df = df.copy()

        df['facing'] = (
            df['facing']
            .fillna('NA')
            .astype(str)
            .str.strip()
        )

        return df

    # ==========================================================
    # DERIVED FEATURES
    # ==========================================================

    def insert_derived_features(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info(
            "HouseCleaner Execution: Creating derived features..."
        )

        df = df.copy()

        df['price'] = pd.to_numeric(
            df['price'],
            errors='coerce'
        )

        df['price_per_sqft'] = pd.to_numeric(
            df['price_per_sqft'],
            errors='coerce'
        )

        calculated_area = round(
            (df['price'] * 10000000)
            / df['price_per_sqft']
        )

        df['area'] = calculated_area

        df.insert(
            loc=1,
            column='property_type',
            value='house'
        )

        return df





