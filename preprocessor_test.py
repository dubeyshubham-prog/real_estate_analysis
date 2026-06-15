import pandas as pd

from src.utils.data_loader import DataLoader
from src.utils.preprocessor import FlatCleaner
from src.utils.preprocessor import HouseCleaner


def test_pipeline():
    print("\n================ LIVE TEST PIPELINE STARTED ================\n")

    # ==========================================================
    # LOAD RAW DATA
    # ==========================================================
    loader = DataLoader()

    flat_df = loader.load_flat_data()
    house_df = loader.load_house_data()

    print(f"[INFO] Flat Raw Data Shape  : {flat_df.shape}")
    print(f"[INFO] House Raw Data Shape : {house_df.shape}")

    # ==========================================================
    # FLAT DATASET PIPELINE
    # ==========================================================
    print("\n================ FLAT PIPELINE =================")

    flat_cleaner = FlatCleaner()

    # STEP 1
    flat_step1 = flat_cleaner.common_drop_and_rename(
        flat_df,
        rename_mapping={
            "area": "price_per_sqft"
        }
    )
    print(f"[TEST] Flat Step 1 Completed | Shape = {flat_step1.shape}")

    # STEP 2
    flat_step2 = flat_cleaner.clean_society(flat_step1)
    print(f"[TEST] Flat Step 2 Completed | Shape = {flat_step2.shape}")

    # STEP 3
    flat_step3 = flat_cleaner.clean_price(flat_step2)
    print(f"\n[TEST] Flat Price Column\n{flat_step3['price'].head()}")

    # STEP 4
    flat_step4 = flat_cleaner.clean_price_per_sqft(flat_step3)
    print(f"\n[TEST] Flat Price Per Sqft\n{flat_step4['price_per_sqft'].head()}")

    # STEP 5
    flat_step5 = flat_cleaner.clean_bedroom(flat_step4)
    print(f"\n[TEST] Flat Bedroom\n{flat_step5['bedRoom'].head()}")

    # STEP 6
    flat_step6 = flat_cleaner.clean_bathroom(flat_step5)
    print(f"\n[TEST] Flat Bathroom\n{flat_step6['bathroom'].head()}")

    # STEP 7
    flat_step7 = flat_cleaner.clean_balcony(flat_step6)
    print(f"\n[TEST] Flat Balcony\n{flat_step7['balcony'].head()}")

    # STEP 8
    flat_step8 = flat_cleaner.clean_additional_room(flat_step7)
    print(f"\n[TEST] Flat Additional Room\n{flat_step8['additionalRoom'].head()}")

    # STEP 9
    flat_step9 = flat_cleaner.clean_floor(flat_step8)
    print(
        f"\n[TEST] Flat Floor Columns\n"
        f"{flat_step9[['floor_num', 'total_floors']].head()}"
    )

    # STEP 10
    flat_step10 = flat_cleaner.clean_facing(flat_step9)
    print(f"\n[TEST] Flat Facing\n{flat_step10['facing'].head()}")

    # STEP 11
    flat_final = flat_cleaner.insert_derived_features(flat_step10)

    print("\n[TEST] Flat Final Columns")
    print(flat_final.columns)

    print("\n[TEST] Flat Final Sample")
    print(flat_final.head(2))

    # ==========================================================
    # HOUSE DATASET PIPELINE
    # ==========================================================
    print("\n================ HOUSE PIPELINE =================")

    house_cleaner = HouseCleaner()

    # STEP 1
    house_step1 = house_cleaner.common_drop_and_rename(
        house_df,
        rename_mapping={
            "rate": "price_per_sqft"
        }
    )

    print(f"[TEST] House Step 1 Completed | Shape = {house_step1.shape}")

    print("\n[DEBUG] House Columns After Rename")
    print(house_step1.columns)

    print("\n[DEBUG] Raw Price Per Sqft Values")
    print(house_step1['price_per_sqft'].head())

    # STEP 2
    house_step2 = house_cleaner.clean_society(house_step1)
    print(f"\n[TEST] House Society\n{house_step2['society'].head()}")

    # STEP 3
    house_step3 = house_cleaner.clean_price(house_step2)
    print(f"\n[TEST] House Price\n{house_step3['price'].head()}")

    # STEP 4
    house_step4 = house_cleaner.clean_price_per_sqft(house_step3)
    print(
        f"\n[TEST] House Price Per Sqft\n"
        f"{house_step4['price_per_sqft'].head()}"
    )

    # STEP 5
    house_step5 = house_cleaner.clean_bedroom(house_step4)
    print(f"\n[TEST] House Bedroom\n{house_step5['bedRoom'].head()}")

    # STEP 6
    house_step6 = house_cleaner.clean_bathroom(house_step5)
    print(f"\n[TEST] House Bathroom\n{house_step6['bathroom'].head()}")

    # STEP 7
    house_step7 = house_cleaner.clean_balcony(house_step6)
    print(f"\n[TEST] House Balcony\n{house_step7['balcony'].head()}")

    # STEP 8
    house_step8 = house_cleaner.clean_additional_room(house_step7)
    print(
        f"\n[TEST] House Additional Room\n"
        f"{house_step8['additionalRoom'].head()}"
    )

    # STEP 9
    house_step9 = house_cleaner.clean_floor(house_step8)

    # STEP 10
    house_step10 = house_cleaner.clean_facing(house_step9)

    # STEP 11
    house_final = house_cleaner.insert_derived_features(house_step10)

    print("\n[TEST] House Final Columns")
    print(house_final.columns)

    print("\n[TEST] House Final Sample")
    print(house_final.head(2))

    print("\n================ TEST PIPELINE SUCCESSFULLY COMPLETED ================\n")

    # ==========================================================
    # MERGE FLAT + HOUSE DATASETS
    # ==========================================================

    house_final["floor_num"] = None
    house_final["total_floors"] = None

    # SAME COLUMN ORDER
    house_df_step11 = house_final[house_final.columns]

    # MERGE BOTH DATASETS
    final_df = pd.concat(
        [flat_final, house_df_step11],
        ignore_index=True
    )

    print("\n================ MERGED DATASET =================")

    print(
        f"[TEST] Final Shape : {final_df.shape}"
    )

    print(
        f"\n[TEST] Property Type Distribution\n"
        f"{final_df['property_type'].value_counts()}"
    )

    print(
        f"\n[TEST] Missing Values\n"
        f"{final_df.isnull().sum().sort_values(ascending=False).head(10)}"
    )

    print(
        f"\n[TEST] Duplicate Rows : "
        f"{final_df.duplicated().sum()}"
    )

    print(
        f"\n[TEST] Final Sample\n"
        f"{final_df.head(5)}"
    )

    final_df.to_csv(
        "data/processed/cleaned_properties.csv",
        index=False
    )


if __name__ == "__main__":
    test_pipeline()