from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

from config.settings import Config
from src.monitoring.logging import get_logger


class RoomClassifierTrainer:
    def __init__(
        self,
        dataset_folder: Path = Config.ROOM_DATASET_DIR,
        batch_size=32,
        image_size=224,
        subset_size=None,
    ):
        self.logger = get_logger(__name__)
        self.dataset_path = Path(dataset_folder)

        self.batch_size = batch_size
        self.image_size = image_size
        self.subset_size = subset_size

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def create_transforms(self):

        self.logger.info("Creating image transformations")

        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])

    def load_dataset(self):

        self.logger.info(
            f"Loading image dataset from: {self.dataset_path}"
        )

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {self.dataset_path}"
            )

        base_dataset = ImageFolder(
            root=self.dataset_path,
            transform=self.create_transforms()
        )

        self.class_names = base_dataset.classes
        self.num_classes = len(self.class_names)

        if self.subset_size:

            self.logger.info(
                f"Using subset dataset with {self.subset_size} images"
            )

            self.dataset = Subset(
                base_dataset,
                range(self.subset_size)
            )

        else:

            self.dataset = base_dataset

        self.logger.info(
            f"Loaded {len(self.dataset)} images"
        )

        self.logger.info(
            f"Detected classes: {self.class_names}"
        )

        return self.dataset

    def create_data_loaders(self):

        self.logger.info("Creating train and validation datasets")

        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size

        train_dataset, val_dataset = random_split(
            self.dataset,
            [train_size, val_size]
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.logger.info(
            f"Train Images: {len(train_dataset)}"
        )

        self.logger.info(
            f"Validation Images: {len(val_dataset)}"
        )

        self.logger.info(
            f"Batch Size: {self.batch_size}"
        )

        return self.train_loader, self.val_loader

    def build_model(self):

        self.logger.info("Loading pretrained ResNet18 model")

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        num_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            num_features,
            self.num_classes
        )

        self.model = self.model.to(self.device)

        self.logger.info(
            f"Final classification layer updated for {self.num_classes} classes"
        )

        self.logger.info(
            f"Using device: {self.device}"
        )

        return self.model

    def setup_training(self):

        self.logger.info("Setting up loss function and optimizer")

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        return self.criterion, self.optimizer

    def prepare(self):

        self.logger.info("Room classifier preparation started")

        self.load_dataset()
        self.create_data_loaders()
        self.build_model()
        self.setup_training()

        self.logger.info("Room classifier preparation completed")

        print("\nDataset Path:", self.dataset_path)
        print("Total Images:", len(self.dataset))
        print("Classes:", self.class_names)
        print("Device:", self.device)
        print("Model Final Layer:", self.model.fc)


if __name__ == "__main__":

    trainer = RoomClassifierTrainer(
        subset_size=500
    )

    trainer.prepare()
