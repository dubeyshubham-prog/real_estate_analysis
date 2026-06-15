import pickle
import numpy as np
import pandas as pd

class PredictionService:
    def __init__(
        self,
        model_path="data/pkl_files/ml_prediction/pipeline.pkl",
        dataframe_path="data/pkl_files/ml_prediction/df.pkl"
    ):
        self.model_path = model_path
        self.dataframe_path = dataframe_path
        self.pipeline = self._load_model()
        self.df = self._load_dataframe()

    # ==========================================================
    # LOAD TRAINED MODEL PIPELINE
    # ==========================================================
    def _load_model(self):
        with open(self.model_path, "rb") as file:
            return pickle.load(file)

    # ==========================================================
    # LOAD REFERENCE DATAFRAME
    # ==========================================================
    def _load_dataframe(self):
        with open(self.dataframe_path, "rb") as file:
            return pickle.load(file)

    # ==========================================================
    # GET DROPDOWN OPTIONS
    # ==========================================================
    def get_form_options(self):
        return {
            "property_types": sorted(self.df["property_type"].unique()),
            "sectors": sorted(self.df["sector"].unique()),
            "balconies": sorted(self.df["balcony"].unique()),
            "age_possession": sorted(self.df["agePossession"].unique()),
            "furnishing_types": sorted(self.df["furnishing_type"].unique()),
            "luxury_categories": sorted(self.df["luxury_category"].unique()),
            "floor_categories": sorted(self.df["floor_category"].unique())
        }

    # ==========================================================
    # PREDICT PROPERTY PRICE
    # ==========================================================
    def predict_price(self, input_data: dict):

        input_df = pd.DataFrame([input_data])

        prediction = self.pipeline.predict(input_df)

        final_price = np.expm1(prediction)[0]

        return round(final_price, 2)