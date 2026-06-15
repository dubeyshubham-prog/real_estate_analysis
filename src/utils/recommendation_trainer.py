import joblib
from src.utils.recommendation import HybridRecommender

class RecommendationTrainer:
    def train(self, apartment_df):
        recommender = HybridRecommender()
        recommender.fit(apartment_df)
        return recommender

    def save_model(
            self,
            recommender,
            save_path
    ):

        joblib.dump(
            recommender,
            save_path
        )

        print(
            f"Recommendation model saved at: {save_path}"
        )