# ==========================================================
# ALL REQUIRED LIBRARIES
# ==========================================================
from src.utils.feature_engineering_level_1 import (AreaFeatureEngineer,
                                                   AdditionalRoomFeatureEngineer,
                                                   AgePossessionFeatureEngineer,
                                                   FurnishingExtractor,
                                                   FurnishingClusterEngineer,
                                                   FeaturesLuxuryScoreEngineer,
                                                   FinalFeatureCleaner)
from src.utils.data_loader import DataLoader

loader = DataLoader()
df = loader.load_gurgaon_property_v1_data()
engineer = AreaFeatureEngineer()

df = engineer.run(df)
print(
    df[
        [
            "super_built_up_area",
            "built_up_area",
            "carpet_area"
        ]
    ].head()
)

engineer = AdditionalRoomFeatureEngineer()

df = engineer.run(df)

print(
    df[
        [
            "additionalRoom",
            "study room",
            "servant room",
            "store room",
            "pooja room",
            "others"
        ]
    ].head(10)
)

df = AgePossessionFeatureEngineer().run(df)

df = FurnishingExtractor().run(df)

print(df.shape)
print(
    df[
        [
            "AC",
            "Fan",
            "Wardrobe",
            "Modular Kitchen"
        ]
    ].head()
)

df = FurnishingClusterEngineer().run(df)
print(df["furnishing_type"].value_counts())
print(df.shape)

apartment = loader.load_apartment_data()

df = FeaturesLuxuryScoreEngineer().run(df, apartment)

print(df[["features", "features_list", "luxury_score"]].head())
print(df["luxury_score"].describe())

df = FinalFeatureCleaner().run(df)

print(df.shape)
print(df.columns)

df.to_csv(
    "data/processed/gurgaon_properties_cleaned_v2.csv",
    index=False
)