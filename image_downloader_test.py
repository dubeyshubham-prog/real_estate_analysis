from src.scraping.image_downloader import ImageDownloader

def test_image_downloader():
    image_url = "https://images.unsplash.com/photo-1560448204-e02f11c3d0e2"

    downloader = ImageDownloader()

    file_path = downloader.download_image(
        image_url=image_url,
        file_name="test_property_image"
    )

    print(file_path)

    assert file_path is not None