import joblib
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]

class RecommendationService:

    def __init__(
            self,
            recommender_path=(
                    BASE_DIR
                    / "data"
                    / "pkl_files"
                    / "recommendation"
                    / "hybrid_recommender.pkl"
            )
    ):
        self.recommender_path = recommender_path
        self.recommender = self._load_recommender()

    # ==========================================================
    # LOAD HYBRID RECOMMENDER
    # ==========================================================
    def _load_recommender(self):

        return joblib.load(
            self.recommender_path
        )

    # ==========================================================
    # GET ALL PROPERTY NAMES
    # ==========================================================
    def get_property_names(self):

        return sorted(
            self.recommender.location_recommender
            .location_df_normalized
            .index
            .tolist()
        )

    # ==========================================================
    # GET RECOMMENDATIONS
    # ==========================================================
    def get_recommendations(
            self,
            property_name: str,
            top_n: int = 5
    ):

        matched_property = (
            self.find_property_name(
                property_name
            )
        )

        if matched_property is None:
            raise ValueError(
                "Property not found"
            )

        result_df = (
            self.recommender.recommend(
                property_name=matched_property,
                top_n=top_n
            )
        )

        return result_df.to_dict(
            orient="records"
        )

    def find_property_name(
            self,
            property_name
    ):

        property_name = (
            property_name
            .strip()
            .lower()
        )

        all_properties = (
            self.get_property_names()
        )

        for item in all_properties:

            if item.lower() == property_name:
                return item

        return None

obj = RecommendationService()