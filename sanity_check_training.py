from src.utils.data_loader import DataLoader
from src.utils.model_selection import SanityCheckTrainer


def test_model_trainer():
    loader = DataLoader()
    
    df = loader.load_feature_selection_data()

    trainer = SanityCheckTrainer()

    X, y = trainer.prepare_data(df)

    pipeline = trainer.create_pipeline()

    pipeline.fit(X, y)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Model training completed")


if __name__ == "__main__":
    test_model_trainer()