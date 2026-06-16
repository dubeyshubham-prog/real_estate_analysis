from src.deep_learning.visual_recommender import VisualEmbeddingDatabase
from src.deep_learning.visual_recommender import (
    VisualSimilaritySearch
)

def test_visual_embedding_database():

    builder = VisualEmbeddingDatabase(
        max_images=None
    )

    database = builder.build_database()

    save_path = builder.save_database(
        database
    )

    print("Total Embeddings:", len(database))
    print("Sample Record Keys:", database[0].keys())
    print("Embedding Shape:", database[0]["embedding"].shape)
    print("Saved At:", save_path)

    assert len(database) > 10000
    assert database[0]["embedding"].shape[0] == 512

def test_visual_similarity_search():
    recommender = VisualSimilaritySearch()

    query_image = r"C:\datasciencejourney\real_estate_analysis\data\property_images\raw\room-dataset\bathroom\bathroom_13.png"

    results = recommender.recommend_similar_images(
        query_image_path=query_image,
        top_k=5
    )

    print("\nTop Recommendations:\n")

    for result in results:

        print(result)

    assert len(results) == 5