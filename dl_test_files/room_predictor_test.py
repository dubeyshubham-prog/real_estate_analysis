from src.deep_learning.predictor import (
    RoomPredictor
)

def test_room_predictor_initialization():

    predictor = RoomPredictor()

    print("Predictor initialized successfully")
    print("Model Path:", predictor.model_path)

    assert predictor.model_path is not None

def test_model_loading():

    predictor = RoomPredictor()

    model = predictor.load_model()

    print("\nModel Loaded Successfully")
    print("Classes:", predictor.class_names)

    assert model is not None

def test_single_image_prediction():

    predictor = RoomPredictor()

    image_path = r"C:\datasciencejourney\real_estate_analysis\data\property_images\raw\room-dataset\bedroom\bedroom_1.jpg"

    result = predictor.predict(
        image_path=image_path
    )

    print("\nPrediction Result:")
    print(result)

    assert "predicted_class" in result
    assert "confidence" in result

test_room_predictor_initialization()
test_model_loading()
test_single_image_prediction()