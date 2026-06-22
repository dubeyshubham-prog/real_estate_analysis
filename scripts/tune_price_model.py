from config.settings import Config
from src.data.pipeline import DataPipeline
from src.ml.tuning import HyperparameterTuner


def main() -> None:
    data = DataPipeline().run().feature_selected
    _, report = HyperparameterTuner().run(data)
    Config.HYPERPARAMETER_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report.to_csv(Config.HYPERPARAMETER_REPORT, index=False)
    print(report.head(10).to_string(index=False))
    print("Saved report:", Config.HYPERPARAMETER_REPORT)


if __name__ == "__main__":
    main()
