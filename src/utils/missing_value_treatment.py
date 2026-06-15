import numpy as np
import pandas as pd


class MissingValueHandler:

    # ==========================================================
    # IMPUTE BUILT-UP AREA USING AREA RATIOS
    # ==========================================================
    def impute_built_up_area(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        all_present_df = df[
            ~(
                df["super_built_up_area"].isnull()
                | df["built_up_area"].isnull()
                | df["carpet_area"].isnull()
            )
        ]

        super_to_built_up_ratio = (
            all_present_df["super_built_up_area"]
            / all_present_df["built_up_area"]
        ).median()

        carpet_to_built_up_ratio = (
            all_present_df["carpet_area"]
            / all_present_df["built_up_area"]
        ).median()

        # super + carpet present, built_up missing
        mask = (
            df["super_built_up_area"].notnull()
            & df["built_up_area"].isnull()
            & df["carpet_area"].notnull()
        )

        df.loc[mask, "built_up_area"] = round(
            (
                (df.loc[mask, "super_built_up_area"] / super_to_built_up_ratio)
                + (df.loc[mask, "carpet_area"] / carpet_to_built_up_ratio)
            ) / 2
        )

        # only super present
        mask = (
            df["super_built_up_area"].notnull()
            & df["built_up_area"].isnull()
            & df["carpet_area"].isnull()
        )

        df.loc[mask, "built_up_area"] = round(
            df.loc[mask, "super_built_up_area"] / super_to_built_up_ratio
        )

        # only carpet present
        mask = (
            df["super_built_up_area"].isnull()
            & df["built_up_area"].isnull()
            & df["carpet_area"].notnull()
        )

        df.loc[mask, "built_up_area"] = round(
            df.loc[mask, "carpet_area"] / carpet_to_built_up_ratio
        )

        return df

    # ==========================================================
    # DROP UNNECESSARY AREA COLUMNS
    # ==========================================================
    def drop_area_columns(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df.drop(
            columns=[
                "area",
                "areaWithType",
                "super_built_up_area",
                "carpet_area",
                "area_room_ratio"
            ],
            inplace=True,
            errors="ignore"
        )

        return df

    # ==========================================================
    # IMPUTE FLOOR NUMBER
    # ==========================================================
    def impute_floor_num(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df["floorNum"] = pd.to_numeric(
            df["floorNum"],
            errors="coerce"
        )

        df["floorNum"] = df["floorNum"].fillna(0.2)

        return df

    # ==========================================================
    # DROP FACING COLUMN
    # ==========================================================
    def drop_facing_column(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df.drop(
            columns=["facing"],
            inplace=True,
            errors="ignore"
        )

        return df

    # ==========================================================
    # DROP ROWS WITH MISSING SOCIETY
    # ==========================================================
    def drop_missing_society_rows(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df = df[df["society"].notnull()]

        return df

    # ==========================================================
    # IMPUTE AGE POSSESSION USING MODE
    # ==========================================================
    def impute_age_possession(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        def fill_by_sector_and_type(row):

            if row["agePossession"] != "Undefined":
                return row["agePossession"]

            mode_value = df[
                (df["sector"] == row["sector"])
                & (df["property_type"] == row["property_type"])
            ]["agePossession"].mode()

            if not mode_value.empty:
                return mode_value.iloc[0]

            return row["agePossession"]

        def fill_by_sector(row):

            if row["agePossession"] != "Undefined":
                return row["agePossession"]

            mode_value = df[
                df["sector"] == row["sector"]
            ]["agePossession"].mode()

            if not mode_value.empty:
                return mode_value.iloc[0]

            return row["agePossession"]

        def fill_by_property_type(row):

            if row["agePossession"] != "Undefined":
                return row["agePossession"]

            mode_value = df[
                df["property_type"] == row["property_type"]
            ]["agePossession"].mode()

            if not mode_value.empty:
                return mode_value.iloc[0]

            return np.nan

        df["agePossession"] = df.apply(fill_by_sector_and_type, axis=1)
        df["agePossession"] = df.apply(fill_by_sector, axis=1)
        df["agePossession"] = df.apply(fill_by_property_type, axis=1)

        return df

    # ==========================================================
    # COMPLETE MISSING VALUE PIPELINE
    # ==========================================================
    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df = self.impute_built_up_area(df)
        df = self.drop_area_columns(df)
        df = self.impute_floor_num(df)
        df = self.drop_facing_column(df)
        df = self.drop_missing_society_rows(df)
        df = self.impute_age_possession(df)

        return df