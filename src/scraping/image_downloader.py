import os
import requests
from urllib.parse import urlparse

class ImageDownloader:
    def __init__(self, save_dir="data/property_images/raw"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _get_file_extension(self, image_url):
        path = urlparse(image_url).path
        extension = os.path.splitext(path)[1]

        if extension.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            return extension

        return ".jpg"

    def download_image(self, image_url, file_name):

        try:
            response = requests.get(
                image_url,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            response.raise_for_status()

            extension = self._get_file_extension(image_url)

            file_path = os.path.join(
                self.save_dir,
                f"{file_name}{extension}"
            )

            with open(file_path, "wb") as file:
                file.write(response.content)

            return file_path

        except Exception as error:
            print(f"Image download failed: {image_url}")
            print(error)
            return None