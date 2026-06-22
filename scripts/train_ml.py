import argparse
import json

from src.data.pipeline import DataPipeline
from src.ml.training import PriceModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data and train the PropLens price model."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and evaluate without saving data or model artifacts.",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save regenerated processed datasets with the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_pipeline = DataPipeline()
    data_result = data_pipeline.run()
    training_result = PriceModelTrainer().run(
        data_result.feature_selected
    )

    print(
        "Training metrics:",
        json.dumps(training_result.metadata["metrics"], indent=2),
    )
    if args.dry_run:
        return

    if args.save_data:
        print("Saved data:", DataPipeline.save(data_result))
    print("Saved model:", PriceModelTrainer.save(training_result))


if __name__ == "__main__":
    main()
