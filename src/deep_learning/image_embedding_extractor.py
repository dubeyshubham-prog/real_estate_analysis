import torch
from torchvision import models
from src.monitoring.logging import get_logger
from torchvision import models, transforms
from PIL import Image

class ImageEmbeddingExtractor:
    def __init__(self):

        self.logger = get_logger(__name__)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def load_feature_extractor(self):

        self.logger.info(
            "Loading ResNet18 feature extractor"
        )

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.model.fc = torch.nn.Identity()

        self.model = self.model.to(
            self.device
        )

        self.model.eval()
        return self.model

    def create_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def extract_embedding(self, image_path):

        if not hasattr(self, "model"):
            self.load_feature_extractor()

        image = Image.open(image_path).convert("RGB")

        transform = self.create_transform()

        image_tensor = transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():

            embedding = self.model(image_tensor)

        embedding = embedding.squeeze(0)

        return embedding.cpu().numpy()
