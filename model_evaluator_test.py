from src.utils.data_loader import DataLoader
from src.utils.model_evaluator import ModelEvaluator


def test_model_evaluator():

    loader = DataLoader()

    df = loader.load_feature_selection_data()

    X = df.drop(columns=["price"])
    y = df["price"]

    evaluator = ModelEvaluator()

    report = evaluator.run(X, y)

    print(report)

    assert not report.empty