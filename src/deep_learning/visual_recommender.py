from pathlib import Path
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.deep_learning.image_embedding_extractor import (
    ImageEmbeddingExtractor
)


class ImageDatabaseBuilder:

    def __init__(
            self,
            image_root="data/property_images/raw/room-dataset"
    ):

        self.project_root = Path(__file__).resolve().parents[2]
        self.image_root = self.project_root / image_root

    def collect_images(self):

        image_records = []

        for folder in self.image_root.iterdir():

            if folder.is_dir():

                for image_path in folder.glob("*"):

                    if image_path.suffix.lower() in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".webp"
                    ]:

                        image_records.append({
                            "image_path": str(image_path),
                            "room_type": folder.name
                        })

        return image_records


class EmbeddingGenerator:

    def __init__(self):

        self.extractor = ImageEmbeddingExtractor()

    def generate_embedding(
            self,
            image_path
    ):

        return self.extractor.extract_embedding(
            image_path
        )


class VisualEmbeddingDatabase:

    def __init__(
            self,
            max_images=None
    ):

        self.image_db_builder = ImageDatabaseBuilder()
        self.embedding_generator = EmbeddingGenerator()
        self.max_images = max_images

    def build_database(self):

        image_records = (
            self.image_db_builder.collect_images()
        )

        if self.max_images is not None:
            image_records = image_records[:self.max_images]

        database = []

        for index, record in enumerate(image_records, start=1):

            embedding = (
                self.embedding_generator.generate_embedding(
                    record["image_path"]
                )
            )

            database.append({
                "image_path": record["image_path"],
                "room_type": record["room_type"],
                "embedding": embedding
            })

            if index % 100 == 0:
                print(f"Processed {index}/{len(image_records)} images")

        return database

    def save_database(
            self,
            database,
            save_path="data/embeddings/visual_embedding_database.pkl"
    ):

        project_root = Path(__file__).resolve().parents[2]
        save_path = project_root / save_path

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(save_path, "wb") as file:
            pickle.dump(
                database,
                file
            )

        return save_path


class VisualSimilaritySearch:

    def __init__(
            self,
            database_path="data/embeddings/visual_embedding_database.pkl"
    ):

        self.project_root = Path(__file__).resolve().parents[2]
        self.database_path = self.project_root / database_path

        self.embedding_generator = EmbeddingGenerator()

    def load_database(self):

        with open(self.database_path, "rb") as file:
            self.database = pickle.load(file)

        self.embeddings = np.array([
            item["embedding"]
            for item in self.database
        ])

        return self.database

    def recommend_similar_images(
            self,
            query_image_path,
            top_k=5
    ):

        if not hasattr(self, "database"):
            self.load_database()

        query_embedding = (
            self.embedding_generator.generate_embedding(
                query_image_path
            )
        )

        query_embedding = query_embedding.reshape(1, -1)

        similarity_scores = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]

        top_indices = similarity_scores.argsort()[::-1][:top_k]

        results = []

        for index in top_indices:

            item = self.database[index]

            relative_path = Path(
                item["image_path"]
            ).relative_to(
                self.project_root /
                "data/property_images/raw/room-dataset"
            )

            image_url = (
                f"/property-images/"
                f"{relative_path.as_posix()}"
            )

            results.append({
                "image_path": item["image_path"],
                "image_url": image_url,
                "room_type": item["room_type"],
                "similarity_score": round(
                    float(similarity_scores[index]),
                    4
                )
            })

        return results