import shutil
from pathlib import Path
from uuid import uuid4

from config.settings import Config


class VisionService:
    def __init__(
        self,
        upload_dir: Path = Config.IMAGE_UPLOADS_DIR,
        database_path: Path = Config.VISUAL_EMBEDDING_DATABASE_FILE,
    ):
        from src.deep_learning.visual_recommender import (
            VisualSimilaritySearch,
        )

        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.visual_search = VisualSimilaritySearch(
            database_path=database_path,
        )

    def save_uploaded_image(self, uploaded_file) -> Path:
        file_extension = Path(uploaded_file.filename or "").suffix.lower()
        unique_file_name = f"{uuid4().hex}{file_extension}"
        save_path = self.upload_dir / unique_file_name

        with save_path.open("wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)

        return save_path

    def find_similar_images(self, uploaded_file, top_k: int = 5) -> dict:
        saved_image_path = self.save_uploaded_image(uploaded_file)
        recommendations = self.visual_search.recommend_similar_images(
            query_image_path=saved_image_path,
            top_k=top_k,
        )

        return {
            "uploaded_image": str(saved_image_path),
            "recommendations": recommendations,
        }
