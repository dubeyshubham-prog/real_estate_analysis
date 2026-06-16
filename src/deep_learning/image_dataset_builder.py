from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
dataset_path = PROJECT_ROOT / "data" / "property_images" / "raw" / "room-dataset"

class ImageDatasetBuilder:

    def __init__(self):
        self.dataset_path = Path(dataset_path)

    def build(self):

        data = []

        for room_type in self.dataset_path.iterdir():

            if room_type.is_dir():

                for image_path in room_type.glob("*"):

                    if image_path.suffix.lower() in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".webp"
                    ]:

                        data.append({
                            "image_path": str(image_path),
                            "room_type": room_type.name
                        })

        df = pd.DataFrame(data)

        return df


if __name__ == "__main__":

    builder = ImageDatasetBuilder()

    df = builder.build()

    print(df.head())

    print("\nTotal Images:", len(df))

    print("\nRoom Distribution:\n")
    print(df["room_type"].value_counts())
