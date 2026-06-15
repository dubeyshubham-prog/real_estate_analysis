# ==========================================================
# ALL REQUIRED LIBRARIES
# ==========================================================
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import ast
from sklearn.preprocessing import MultiLabelBinarizer

class AreaFeatureEngineer:
    # ==========================================================
    # EXTRACT SUPER BUILT-UP AREA
    # ==========================================================
    def _super_built_up_area(self, text):

        if pd.isna(text):
            return None

        match = re.search(
            r"Super Built up area (\d+\.?\d*)",
            str(text)
        )

        if match:
            return float(match.group(1))

        return None

    # ==========================================================
    # EXTRACT BUILT-UP / CARPET AREA
    # ==========================================================
    def _get_area(self, text, area_type):

        if pd.isna(text):
            return None

        match = re.search(
            area_type + r"\s*:\s*(\d+\.?\d*)",
            str(text)
        )

        if match:
            return float(match.group(1))

        return None

    # ==========================================================
    # CONVERT SQ.M TO SQ.FT
    # ==========================================================
    def _convert_to_sqft(self, text, area_value):

        if area_value is None or pd.isna(text):
            return area_value

        match = re.search(
            rf"{area_value} \((\d+\.?\d*) sq.m.\)",
            str(text)
        )

        if match:
            sq_m_value = float(match.group(1))
            return sq_m_value * 10.7639

        return area_value

    # ==========================================================
    # EXTRACT PLOT AREA
    # ==========================================================
    def _extract_plot_area(self, text):

        if pd.isna(text):
            return None

        match = re.search(
            r"Plot area (\d+\.?\d*)",
            str(text)
        )

        if match:
            return float(match.group(1))

        return None

    # ==========================================================
    # FIX AREA SCALE
    # ==========================================================
    def _convert_scale(self, row):

        if (
                pd.isna(row["area"])
                or pd.isna(row["built_up_area"])
        ):
            return row["built_up_area"]

        ratio = round(
            row["area"] / row["built_up_area"]
        )

        if ratio == 9:
            return row["built_up_area"] * 9

        elif ratio == 11:
            return row["built_up_area"] * 10.7

        return row["built_up_area"]

    # ==========================================================
    # MAIN EXECUTION
    # ==========================================================
    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        # -------------------------
        # SUPER BUILT-UP AREA
        # -------------------------
        df["super_built_up_area"] = (
            df["areaWithType"]
            .apply(self._super_built_up_area)
        )

        df["super_built_up_area"] = df.apply(
            lambda row: self._convert_to_sqft(
                row["areaWithType"],
                row["super_built_up_area"]
            ),
            axis=1
        )

        # -------------------------
        # BUILT-UP AREA
        # -------------------------
        df["built_up_area"] = (
            df["areaWithType"]
            .apply(
                lambda x: self._get_area(
                    x,
                    "Built Up area"
                )
            )
        )

        df["built_up_area"] = df.apply(
            lambda row: self._convert_to_sqft(
                row["areaWithType"],
                row["built_up_area"]
            ),
            axis=1
        )

        # -------------------------
        # CARPET AREA
        # -------------------------
        df["carpet_area"] = (
            df["areaWithType"]
            .apply(
                lambda x: self._get_area(
                    x,
                    "Carpet area"
                )
            )
        )

        df["carpet_area"] = df.apply(
            lambda row: self._convert_to_sqft(
                row["areaWithType"],
                row["carpet_area"]
            ),
            axis=1
        )

        # -------------------------
        # HANDLE ALL NAN ROWS
        # -------------------------
        mask = (
                df["super_built_up_area"].isnull()
                & df["built_up_area"].isnull()
                & df["carpet_area"].isnull()
        )

        temp_df = df.loc[
            mask,
            [
                "area",
                "areaWithType",
                "built_up_area"
            ]
        ].copy()

        temp_df["built_up_area"] = (
            temp_df["areaWithType"]
            .apply(self._extract_plot_area)
        )

        temp_df["built_up_area"] = temp_df.apply(
            self._convert_scale,
            axis=1
        )
        df.update(temp_df)
        return df

class AdditionalRoomFeatureEngineer:

    # ==========================================================
    # CREATE BINARY FEATURES FROM additionalRoom
    # ==========================================================
    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        room_columns = [
            "study room",
            "servant room",
            "store room",
            "pooja room",
            "others"
        ]

        for room in room_columns:

            df[room] = (
                df["additionalRoom"]
                .fillna("")
                .str.contains(room, case=False)
                .astype(int)
            )

        return df

class AgePossessionFeatureEngineer:
    # ==========================================================
    # CATEGORIZE AGE POSSESSION
    # ==========================================================
    def _categorize_age_possession(self, value):

        if pd.isna(value):
            return "Undefined"

        value = str(value)

        if (
            "0 to 1 Year Old" in value
            or "Within 6 months" in value
            or "Within 3 months" in value
        ):
            return "New Property"

        if "1 to 5 Year Old" in value:
            return "Moderately Old"

        if "10+ Year Old" in value:
            return "Old Property"

        if (
            "Under Construction" in value
            or "By" in value
        ):
            return "Under Construction"

        try:
            int(value.split(" ")[-1])
            return "Under Construction"

        except:
            return "Undefined"

    # ==========================================================
    # MAIN EXECUTION
    # ==========================================================
    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df["agePossession"] = (
            df["agePossession"]
            .apply(self._categorize_age_possession)
        )

        return df

class FurnishingExtractor:

    # ==========================================================
    # EXTRACT COUNT OF A FURNISHING
    # ==========================================================
    def _get_furnishing_count(
        self,
        details,
        furnishing
    ):

        if not isinstance(details, str):
            return 0

        if f"No {furnishing}" in details:
            return 0

        pattern = re.compile(
            rf"(\d+)\s+{re.escape(furnishing)}"
        )

        match = pattern.search(details)

        if match:
            return int(match.group(1))

        elif furnishing in details:
            return 1

        return 0

    # ==========================================================
    # MAIN EXECUTION
    # ==========================================================
    def run(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        df = df.copy()

        all_furnishings = []

        for detail in df["furnishDetails"].dropna():

            furnishings = (
                detail
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .split(", ")
            )

            all_furnishings.extend(furnishings)

        unique_furnishings = list(
            set(all_furnishings)
        )

        columns_to_include = [
            re.sub(
                r"No |\d+",
                "",
                item
            ).strip()
            for item in unique_furnishings
        ]

        columns_to_include = list(
            set(columns_to_include)
        )

        columns_to_include = [
            col
            for col in columns_to_include
            if col
        ]

        for furnishing in columns_to_include:

            df[furnishing] = (
                df["furnishDetails"]
                .apply(
                    lambda x:
                    self._get_furnishing_count(
                        x,
                        furnishing
                    )
                )
            )

        return df

class FurnishingClusterEngineer:
    # ==========================================================
    # CREATE FURNISHING TYPE USING KMEANS CLUSTERING
    # ==========================================================
    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        furnishing_columns = [
            col for col in df.columns
            if col not in [
                "property_type", "society", "sector", "price",
                "price_per_sqft", "area", "areaWithType",
                "bedRoom", "bathroom", "balcony", "additionalRoom",
                "address", "floorNum", "facing", "agePossession",
                "nearbyLocations", "furnishDetails", "features",
                "floor_num", "total_floors",
                "super_built_up_area", "built_up_area", "carpet_area",
                "study room", "servant room", "store room",
                "pooja room", "others"
            ]
        ]

        scaler = StandardScaler()

        scaled_data = scaler.fit_transform(
            df[furnishing_columns]
        )

        kmeans = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        )

        df["furnishing_type"] = kmeans.fit_predict(
            scaled_data
        )

        df.drop(
            columns=furnishing_columns,
            inplace=True
        )

        return df

class FeaturesLuxuryScoreEngineer:
    # ==========================================================
    # LUXURY FEATURE WEIGHTS
    # ==========================================================
    def __init__(self):
        self.weights = {
            '24/7 Power Backup': 8,
            '24/7 Water Supply': 4,
            '24x7 Security': 7,
            'ATM': 4,
            'Aerobics Centre': 6,
            'Airy Rooms': 8,
            'Amphitheatre': 7,
            'Badminton Court': 7,
            'Banquet Hall': 8,
            'Bar/Chill-Out Lounge': 9,
            'Barbecue': 7,
            'Basketball Court': 7,
            'Business Lounge': 9,
            'CCTV Camera Security': 8,
            'Car Parking': 6,
            'Club House': 9,
            'Concierge Service': 9,
            'Fire Fighting Systems': 8,
            'Fitness Centre / GYM': 8,
            'Flower Garden': 7,
            'Gated Community': 7,
            'Golf Course': 10,
            'Gymnasium': 8,
            'High Speed Elevators': 8,
            'Intercom Facility': 7,
            'Jogging Track': 7,
            'Landscape Garden': 8,
            'Maintenance Staff': 6,
            'Park': 8,
            'Piped Gas': 7,
            'Power Back up Lift': 8,
            'Private Garden / Terrace': 9,
            'Security Personnel': 9,
            'Shopping Centre': 7,
            'Swimming Pool': 8,
            'Visitor Parking': 7,
            'Water Storage': 7,
            'Yoga/Meditation Area': 7
        }

    # ==========================================================
    # FILL MISSING FEATURES FROM APARTMENT DATA
    # ==========================================================
    def fill_missing_features(self, df, apartment_df):

        df = df.copy()
        apartment_df = apartment_df.copy()

        apartment_df["PropertyName"] = (
            apartment_df["PropertyName"]
            .str.lower()
        )

        missing_df = df[df["features"].isnull()]

        matched_features = (
            missing_df
            .merge(
                apartment_df,
                left_on="society",
                right_on="PropertyName",
                how="left"
            )["TopFacilities"]
        )

        df.loc[missing_df.index, "features"] = matched_features.values

        return df

    # ==========================================================
    # CONVERT FEATURES STRING TO LIST
    # ==========================================================
    def create_features_list(self, df):

        df = df.copy()

        df["features_list"] = df["features"].apply(
            lambda x: ast.literal_eval(x)
            if pd.notnull(x) and str(x).startswith("[")
            else []
        )

        return df

    # ==========================================================
    # CREATE LUXURY SCORE
    # ==========================================================
    def create_luxury_score(self, df):

        df = df.copy()

        mlb = MultiLabelBinarizer()

        features_binary_matrix = mlb.fit_transform(
            df["features_list"]
        )

        features_binary_df = pd.DataFrame(
            features_binary_matrix,
            columns=mlb.classes_,
            index=df.index
        )

        available_weights = {
            feature: weight
            for feature, weight in self.weights.items()
            if feature in features_binary_df.columns
        }

        df["luxury_score"] = (
            features_binary_df[list(available_weights.keys())]
            .multiply(list(available_weights.values()))
            .sum(axis=1)
        )

        return df

    # ==========================================================
    # MAIN EXECUTION
    # ==========================================================
    def run(self, df, apartment_df):

        df = self.fill_missing_features(df, apartment_df)
        df = self.create_features_list(df)
        df = self.create_luxury_score(df)
        return df

class FinalFeatureCleaner:
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.drop(
            columns=[
                "nearbyLocations",
                "furnishDetails",
                "features",
                "features_list",
                "additionalRoom",
                "address"
            ],
            inplace=True,
            errors="ignore"
        )
        return df

