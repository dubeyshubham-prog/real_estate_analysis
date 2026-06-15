from src.utils.data_loader import DataLoader
from src.utils.recommendation import FacilityRecommender
from src.utils.data_loader import DataLoader
from src.utils.recommendation import PriceConfigRecommender
from src.utils.data_loader import DataLoader
from src.utils.recommendation import LocationRecommender
from src.utils.recommendation import HybridRecommender

def test_facility_recommender():

    loader = DataLoader()

    df = loader.load_apartment_data()

    recommender = FacilityRecommender()

    recommender.fit(df)

    result = recommender.recommend(
        property_name="DLF The Arbour",
        top_n=5
    )

    print(result)

    assert len(result) == 5

def test_price_config_recommender():

    loader = DataLoader()

    df = loader.load_apartment_data()

    recommender = PriceConfigRecommender()

    recommender.fit(df)

    result = recommender.recommend(
        property_name="M3M Golf Hills",
        top_n=5
    )

    print(result)

    assert len(result) == 5

def test_location_recommender():

    loader = DataLoader()

    df = loader.load_apartment_data()

    recommender = LocationRecommender()

    recommender.fit(df)

    result = recommender.recommend(
        property_name="Ireo Victory Valley",
        top_n=5
    )

    print(result)

    assert len(result) == 5

def test_hybrid_recommender():

    loader = DataLoader()

    df = loader.load_apartment_data()

    recommender = HybridRecommender()

    recommender.fit(df)

    result = recommender.recommend(
        property_name="Ireo Victory Valley",
        top_n=5
    )

    print(result)

    assert len(result) == 5

from src.utils.data_loader import DataLoader
from src.utils.recommendation_trainer import RecommendationTrainer


def test_recommendation_trainer():

    loader = DataLoader()

    apartment_df = loader.load_apartment_data()

    trainer = RecommendationTrainer()

    recommender = trainer.train(
        apartment_df
    )

    trainer.save_model(
        recommender,
        "data/pkl_files/recommendation/hybrid_recommender.pkl"
    )

    assert recommender is not None