import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.recommendation.common import parse_literal
from src.recommendation.common import top_similar
from src.recommendation.common import validate_recommendation_data


class FacilityRecommender:
    """Recommend projects using TF-IDF similarity over facilities."""

    def fit(self, dataframe: pd.DataFrame) -> "FacilityRecommender":
        self.dataframe = validate_recommendation_data(dataframe)
        facilities = self.dataframe["TopFacilities"].map(
            lambda value: parse_literal(value, list)
        )
        facility_text = facilities.map(
            lambda values: " ".join(str(value) for value in values)
        )
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
        )
        matrix = self.vectorizer.fit_transform(facility_text)
        self.similarity_matrix = cosine_similarity(matrix)
        self.property_names = pd.Index(self.dataframe["PropertyName"])
        return self

    def recommend(
        self,
        property_name: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        return top_similar(
            self.property_names,
            self.similarity_matrix,
            property_name,
            top_n,
        )
