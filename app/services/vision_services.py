from pathlib import Path
import shutil
from uuid import uuid4
from src.deep_learning.visual_recommender import VisualSimilaritySearch

class VisionService:

    def __init__(
            self,
            upload_dir="data/uploaded_images"
    ):

        self.project_root = Path(__file__).resolve().parents[2]

        self.upload_dir = (
            self.project_root / upload_dir
        )

        self.upload_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.visual_search = VisualSimilaritySearch()

    def save_uploaded_image(self, uploaded_file):

        file_extension = Path(
            uploaded_file.filename
        ).suffix

        unique_file_name = (
            f"{uuid4().hex}{file_extension}"
        )

        save_path = (
            self.upload_dir / unique_file_name
        )

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(
                uploaded_file.file,
                buffer
            )

        return save_path

    def find_similar_images(
            self,
            uploaded_file,
            top_k=5
    ):

        saved_image_path = (
            self.save_uploaded_image(
                uploaded_file
            )
        )

        recommendations = (
            self.visual_search.recommend_similar_images(
                query_image_path=str(
                    saved_image_path
                ),
                top_k=top_k
            )
        )

        return {
            "uploaded_image": str(
                saved_image_path
            ),
            "recommendations": recommendations
        }