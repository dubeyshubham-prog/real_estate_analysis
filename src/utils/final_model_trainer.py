import pickle
import numpy as np
import pandas as pd
import os
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder
)


class FinalModelTrainer:

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

        pipeline = Pipeline([
            (
                "preprocessor",
                self.create_preprocessor()
            ),
            (
                "regressor",
                XGBRegressor(
                    learning_rate=0.1,
                    max_depth=5,
                    n_estimators=200,
                    subsample=0.8,
                    random_state=42,
                    objective="reg:squarederror"
                )
            )
        ])

        return pipeline

    # ==========================================================
    # TRAIN FINAL MODEL
    # ==========================================================
    def train(self, df: pd.DataFrame):

        X, y = self.prepare_data(df)

        pipeline = self.create_pipeline()

        pipeline.fit(X, y)

        return pipeline, X

    # ==========================================================
    # SAVE MODEL ARTIFACTS
    # ==========================================================
    def save_artifacts(
            self,
            pipeline,
            X,
            model_path="data/pkl_files/ml_prediction/pipeline.pkl",
            dataframe_path="data/pkl_files/ml_prediction/df.pkl"
    ):
        os.makedirs(
            os.path.dirname(model_path),
            exist_ok=True
        )

        os.makedirs(
            os.path.dirname(dataframe_path),
            exist_ok=True
        )

        with open(model_path, "wb") as file:
            pickle.dump(pipeline, file)

        with open(dataframe_path, "wb") as file:
            pickle.dump(X, file)

        print("Model saved at:", model_path)
        print("DataFrame saved at:", dataframe_path)

    # ==========================================================
    # COMPLETE FINAL MODEL PIPELINE
    # ==========================================================
    def run(self, df: pd.DataFrame):

        pipeline, X = self.train(df)

        self.save_artifacts(
            pipeline=pipeline,
            X=X
        )

        return pipeline, X