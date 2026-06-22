import pandas as pd

from src.preprocessing.outliers import OutlierTreatmentPipeline


def test_price_per_sqft_recalculation_handles_zero_area() -> None:
    dataframe = pd.DataFrame(
        {
            "price": [1.0, 2.0],
            "area": [1000.0, 0.0],
        }
    )

    result = OutlierTreatmentPipeline().recalculate_price_per_sqft(dataframe)

    assert result["price_per_sqft"].tolist() == [10_000.0]


def test_bedroom_outliers_remove_invalid_counts() -> None:
    dataframe = pd.DataFrame({"bedRoom": [0.0, 2.0, 11.0]})

    result = OutlierTreatmentPipeline().treat_bedroom_outliers(dataframe)

    assert result["bedRoom"].tolist() == [2.0]
