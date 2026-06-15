from src.utils.data_loader import DataLoader
from src.utils.hyperparameter_tuning import HyperparameterTuner

def test_hyperparameter_tuner():

    loader = DataLoader()

    df = loader.load_feature_selection_data()

    X = df.drop(columns=["price"])
    y = df["price"]

    tuner = HyperparameterTuner()

    result_df = tuner.run(X, y)

    print(result_df[["model", "best_score", "best_params"]])

    assert len(result_df) > 0