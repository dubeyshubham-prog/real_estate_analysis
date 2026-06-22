from config.settings import Config
from src.data.pipeline import DataPipeline
from src.ml.evaluation import ModelEvaluator


def main() -> None:
    data = DataPipeline().run().feature_selected
    report = ModelEvaluator().run(data)
    Config.MODEL_EVALUATION_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report.to_csv(Config.MODEL_EVALUATION_REPORT, index=False)
    print(report.to_string(index=False))
    print("Saved report:", Config.MODEL_EVALUATION_REPORT)


if __name__ == "__main__":
    main()
