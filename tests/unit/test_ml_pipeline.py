from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.ml.evaluation import ModelEvaluator
from src.ml.inference import PricePredictor
from src.ml.preprocessing import PriceModelData
from src.ml.training import PriceModelTrainer


def sample_training_data(rows: int = 30) -> pd.DataFrame:
    records = []
    for index in range(rows):
        records.append(
            {
                "property_type": "flat" if index % 2 == 0 else "house",
                "sector": f"sector {(index % 3) + 1}",
                "bedRoom": float((index % 4) + 1),
                "bathroom": float((index % 3) + 1),
                "balcony": float(index % 3),
                "agePossession": "New Property"
                if index % 2 == 0
                else "Moderately Old",
                "floor_num": float(index % 10),
                "total_floors": float((index % 10) + 5),
                "built_up_area": float(800 + index * 40),
                "servant room": index % 2,
                "store room": (index + 1) % 2,
                "furnishing_type": index % 3,
                "luxury_category": ["Low", "Medium", "High"][index % 3],
                "floor_category": ["Low Floor", "Mid Floor", "High Floor"][
                    index % 3
                ],
                "price": float(0.5 + index * 0.08),
            }
        )
    return pd.DataFrame(records)


def test_model_data_maps_furnishing_labels() -> None:
    features, target = PriceModelData.split_features_target(
        sample_training_data()
    )

    assert set(features["furnishing_type"]) == {
        "unfurnished",
        "semifurnished",
        "furnished",
    }
    assert len(features) == len(target)


def test_model_evaluation_uses_complete_pipeline() -> None:
    evaluator = ModelEvaluator(
        models={
            "RandomForest": RandomForestRegressor(
                n_estimators=10,
                random_state=42,
                n_jobs=1,
            )
        },
        cv_splits=2,
    )

    report = evaluator.run(sample_training_data())

    assert report["model"].tolist() == ["RandomForest"]
    assert report["mean_log_r2"].notna().all()


def test_training_artifacts_support_inference(tmp_path: Path) -> None:
    trainer = PriceModelTrainer(
        model_params={
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.1,
            "subsample": 1.0,
        }
    )
    result = trainer.run(sample_training_data())
    paths = trainer.save(
        result,
        model_path=tmp_path / "model.joblib",
        reference_path=tmp_path / "reference.joblib",
        metadata_path=tmp_path / "metadata.json",
    )
    predictor = PricePredictor(
        model_path=paths["model"],
        reference_path=paths["reference_data"],
        metadata_path=paths["metadata"],
    )
    input_data = result.reference_data.iloc[0].to_dict()

    prediction = predictor.predict(input_data)

    assert prediction["predicted_price_cr"] > 0
    assert prediction["lower_estimate_cr"] <= prediction["upper_estimate_cr"]
