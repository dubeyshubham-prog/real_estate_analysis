from src.utils.data_loader import DataLoader
from src.utils.feature_selection import FeatureSelector


def test_feature_selection():

    loader = DataLoader()

    df = loader.load_missing_value_data()

    print("Before:", df.shape)

    selector = FeatureSelector()

    export_df, fi_report, score_all, score_selected = selector.run(df)

    print("Score with all features:", score_all)
    print("Score after dropping columns:", score_selected)

    print("\nFinal Shape:", export_df.shape)
    print(export_df.columns)

    print("\nFeature Importance Report:")
    print(fi_report)

    export_df.to_csv(
        "data/processed/gurgaon_properties_post_feature_selection.csv",
        index=False
    )

    print("Feature selected dataset saved successfully")


if __name__ == "__main__":
    test_feature_selection()