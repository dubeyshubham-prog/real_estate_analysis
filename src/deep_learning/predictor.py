from pathlib import Path
import torch
from torchvision import transforms, models
from PIL import Image
from src.utils.logger import get_logger

class RoomPredictor:
    def __init__(
            self,
            model_path="data/pkl_files/deep_learning/room_classifier_test.pth"
    ):

        self.logger = get_logger(__name__)

        self.project_root = Path(__file__).resolve().parents[2]

        self.model_path = (
                self.project_root /
                model_path
        )

    def load_model(self):
        self.logger.info(
            f"Loading room classifier model from: {self.model_path}"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location="cpu"
        )

        self.class_names = checkpoint["class_names"]

        self.model = models.resnet18(
            weights=None
        )

        num_features = self.model.fc.in_features

        self.model.fc = torch.nn.Linear(
            num_features,
            len(self.class_names)
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.logger.info(
            f"Model loaded successfully with classes: {self.class_names}"
        )
        return self.model

    def create_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def predict(self, image_path):

        if not hasattr(self, "model"):
            self.load_model()

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        transform = self.create_transform()

        image_tensor = transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():

            outputs = self.model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_index = torch.max(
                probabilities,
                1
            )

        predicted_class = self.class_names[
            predicted_index.item()
        ]

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence.item(), 4)
        }