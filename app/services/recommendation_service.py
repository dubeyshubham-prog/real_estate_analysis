import joblib

class RecommendationService:

    def __init__(
            self,
            recommender_path="data/pkl_files/recommendation/hybrid_recommender.pkl"
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

        result_df = self.recommender.recommend(
            property_name=property_name,
            top_n=top_n
        )

        return result_df.to_dict(
            orient="records"
        )