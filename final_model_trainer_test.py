import numpy as np
import pandas as pd

from src.utils.data_loader import DataLoader
from src.utils.final_model_trainer import FinalModelTrainer


def test_final_model_trainer():

    loader = DataLoader()

    df = loader.load_feature_selection_data()

    trainer = FinalModelTrainer()

    pipeline, X = trainer.run(df)

    sample = X.iloc[[0]]

    prediction = np.expm1(
        pipeline.predict(sample)
    )

    print("Sample Input:")
    print(sample)

    print("\nPredicted Price:")
    print(prediction)

    print("\nFinal model and dataframe saved successfully")

    assert prediction[0] > 0