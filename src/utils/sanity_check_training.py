# ==========================================================
#ALL REQUIRED LIBRARIES:
# ==========================================================
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

class SanityCheckTrainer:
    # ==========================================================
    # PREPARE FEATURES AND TARGET
    # ==========================================================
    def prepare_data(self, df: pd.DataFrame):

        df = df.copy()

        df["furnishing_type"] = df["furnishing_type"].replace({
            0: "unfurnished",
            1: "semifurnished",
            2: "furnished"
        })

        X = df.drop(columns=["price"])
        y = np.log1p(df["price"])

        return X, y

    # ==========================================================
    # CREATE PREPROCESSOR
    # ==========================================================
    def create_preprocessor(self):

        columns_to_encode = [
            "property_type",
            "balcony",
            "furnishing_type",
            "luxury_category",
            "floor_category"
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    [
                        "bedRoom",
                        "bathroom",
                        "built_up_area",
                        "servant room",
                        "store room"
                    ]
                ),
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1
                    ),
                    columns_to_encode
                ),
                (
                    "cat1",
                    OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        handle_unknown="ignore"
                    ),
                    [
                        "sector",
                        "agePossession"
                    ]
                )
            ],
            remainder="passthrough"
        )

        return preprocessor

    # ==========================================================
    # CREATE FINAL MODEL PIPELINE
    # ==========================================================
    def create_pipeline(self):

        preprocessor = self.create_preprocessor()

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=50,
                random_state=42
            ))
        ])

        return pipeline