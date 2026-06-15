import pandas as pd

class OutlierDetector:
    # ==========================================================
    # PRICE PER SQFT OUTLIER TREATMENT
    # ==========================================================
    def treat_price_per_sqft_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        q1 = df["price_per_sqft"].quantile(0.25)
        q3 = df["price_per_sqft"].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df[
            (df["price_per_sqft"] < lower_bound)
            | (df["price_per_sqft"] > upper_bound)
        ].copy()

        outliers["area"] = outliers["area"].apply(
            lambda x: x * 9 if x < 1000 else x
        )

        outliers["price_per_sqft"] = round(
            (outliers["price"] * 10000000) / outliers["area"]
        )

        df.update(outliers)

        df = df[df["price_per_sqft"] <= 50000]

        return df

    # ==========================================================
    # AREA OUTLIER TREATMENT
    # ==========================================================
    def treat_area_outliers(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df = df[df["area"] < 100000]

        indexes_to_drop = [
            2679, 3488, 181, 3430, 1144, 1324,
            3071, 2513, 853, 589, 495, 545,
            1474, 1842, 3061, 1838, 3135, 1103
        ]

        df.drop(
            index=indexes_to_drop,
            inplace=True,
            errors="ignore"
        )

        return df

    # ==========================================================
    # BEDROOM OUTLIER TREATMENT
    # ==========================================================
    def treat_bedroom_outliers(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df = df[df["bedRoom"] <= 10]

        return df

    # ==========================================================
    # RECALCULATE PRICE PER SQFT
    # ==========================================================
    def recalculate_price_per_sqft(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df["price_per_sqft"] = round(
            (df["price"] * 10000000) / df["area"]
        )

        return df

    # ==========================================================
    # AREA ROOM RATIO OUTLIER TREATMENT
    # ==========================================================
    # ==========================================================
    # AREA ROOM RATIO OUTLIER TREATMENT
    # ==========================================================
    def treat_area_room_ratio_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # CONVERT IMPORTANT COLUMNS TO NUMERIC
        df["area"] = pd.to_numeric(df["area"], errors="coerce")
        df["bedRoom"] = pd.to_numeric(df["bedRoom"], errors="coerce")
        df["floorNum"] = pd.to_numeric(df["floorNum"], errors="coerce")

        df["area_room_ratio"] = df["area"] / df["bedRoom"]

        df = df[df["area_room_ratio"] > 100]

        outliers_df = df[
            (df["area_room_ratio"] < 250)
            & (df["bedRoom"] > 3)
            & (df["floorNum"].notnull())
            & (df["floorNum"] != 0)
            ].copy()

        outliers_df["bedRoom"] = round(
            outliers_df["bedRoom"] / outliers_df["floorNum"]
        )

        df.update(outliers_df)

        df["area_room_ratio"] = df["area"] / df["bedRoom"]

        df = df[
            ~(
                    (df["area_room_ratio"] < 250)
                    & (df["bedRoom"] > 4)
            )
        ]

        return df

    # ==========================================================
    # COMPLETE OUTLIER DETECTION PIPELINE
    # ==========================================================
    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        df = df.reset_index(drop=True)

        df = self.treat_price_per_sqft_outliers(df)
        df = self.treat_area_outliers(df)
        df = self.treat_bedroom_outliers(df)
        df = self.recalculate_price_per_sqft(df)
        df = self.treat_area_room_ratio_outliers(df)

        return df