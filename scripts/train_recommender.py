import argparse

from src.data.loader import DataLoader
from src.recommendation.training import RecommendationTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the PropLens hybrid recommendation model."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and validate without saving the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apartment_data = DataLoader().load_apartment_data()
    trainer = RecommendationTrainer()
    recommender = trainer.train(apartment_data)

    sample_property = recommender.property_names[0]
    sample_results = recommender.recommend(sample_property, top_n=3)
    print("Projects:", len(recommender.property_names))
    print("Sample property:", sample_property)
    print(sample_results.to_string(index=False))

    if not args.dry_run:
        print("Saved artifacts:", trainer.save(recommender))


if __name__ == "__main__":
    main()
