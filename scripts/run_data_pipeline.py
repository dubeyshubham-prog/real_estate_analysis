import argparse

from src.data.pipeline import DataPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete PropLens tabular-data pipeline."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run and validate all stages without overwriting processed files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = DataPipeline()
    result = pipeline.run()

    print(
        "Pipeline complete:",
        {
            "rows": len(result.feature_selected),
            "selected_features": result.feature_selected.shape[1] - 1,
            "score_all": round(result.score_all_features, 4),
            "score_selected": round(result.score_selected_features, 4),
        },
    )

    if not args.dry_run:
        saved_paths = pipeline.save(result)
        printable_paths = {
            name: str(path)
            for name, path in saved_paths.items()
        }
        print("Saved outputs:", printable_paths)


if __name__ == "__main__":
    main()
