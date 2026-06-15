from src.utils.data_loader import DataLoader
from src.utils.preprocessing_level_2 import PreprocessorLevel2


def test_feature_engineering():

    loader = DataLoader()

    df = loader.load_cleaned_property_data()

    cleaner = PreprocessorLevel2()

    cleaned_df = cleaner.run_pipeline(df)

    print(cleaned_df.head())

    print("\nShape:", cleaned_df.shape)

    print("\nColumns:")
    print(cleaned_df.columns)

    print("\nMissing Values:")
    print(cleaned_df.isnull().sum().sort_values(ascending=False).head(10))

    # cleaned_df.to_csv(
    #     "data/processed/gurgaon_properties_cleaned_v1.csv",
    #     index=False
    # )
    print(cleaned_df.shape)

if __name__ == "__main__":
    test_feature_engineering()