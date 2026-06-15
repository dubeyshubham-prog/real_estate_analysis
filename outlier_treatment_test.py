import pandas as pd

from src.utils.outlier_treatment import OutlierDetector
from src.utils.data_loader import DataLoader


def test_outlier_detector():
    loader = DataLoader()
    df = loader.load_gurgaon_property_v2_data()

    print("Before:", df.shape)

    detector = OutlierDetector()

    df = detector.run(df)

    print("After:", df.shape)

    print(df[["price", "area", "price_per_sqft", "bedRoom", "area_room_ratio"]].describe())

    df.to_csv(
        "data/processed/gurgaon_properties_outlier_treated.csv",
        index=False
    )

    print("Outlier treated dataset saved successfully")


if __name__ == "__main__":
    test_outlier_detector()