from src.utils.data_loader import DataLoader
from src.utils.missing_value_treatment import MissingValueHandler


def test_missing_value_handler():

    loader = DataLoader()

    df = loader.load_outlier_treated_data()

    print("Before:", df.shape)
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    handler = MissingValueHandler()

    df = handler.run(df)

    print("After:", df.shape)
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    df.to_csv(
        "data/processed/gurgaon_properties_missing_value_imputation.csv",
        index=False
    )

    print("Missing value treated dataset saved successfully")


if __name__ == "__main__":
    test_missing_value_handler()