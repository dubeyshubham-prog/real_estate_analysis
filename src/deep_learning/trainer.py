import torch
from src.utils.logger import get_logger
from pathlib import Path
import torch

class TrainingMetrics:
    def calculate_accuracy(self, outputs, labels):

        _, predictions = outputs.max(1)

        correct_predictions = (
            predictions == labels
        ).sum().item()

        total_predictions = labels.size(0)

        accuracy = correct_predictions / total_predictions

        return accuracy


class SingleEpochTrainer:
    def __init__(self, model, criterion, optimizer, device):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.metrics = TrainingMetrics()

    def train_one_batch(self, images, labels):

        images = images.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.model(images)

        loss = self.criterion(
            outputs,
            labels
        )

        loss.backward()

        self.optimizer.step()

        accuracy = self.metrics.calculate_accuracy(
            outputs,
            labels
        )

        return loss.item(), accuracy

    def train_one_epoch(self, train_loader):
        self.model.train()

        total_loss = 0
        total_accuracy = 0

        for images, labels in train_loader:

            batch_loss, batch_accuracy = self.train_one_batch(
                images,
                labels
            )

            total_loss += batch_loss
            total_accuracy += batch_accuracy

        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = total_accuracy / len(train_loader)

        return epoch_loss, epoch_accuracy

class ValidationEvaluator:

    def __init__(self, model, criterion, device):

        self.model = model
        self.criterion = criterion
        self.device = device
        self.metrics = TrainingMetrics()

    def validate(self, val_loader):

        self.model.eval()

        total_loss = 0
        total_accuracy = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels
                )

                accuracy = self.metrics.calculate_accuracy(
                    outputs,
                    labels
                )

                total_loss += loss.item()
                total_accuracy += accuracy

        val_loss = total_loss / len(val_loader)
        val_accuracy = total_accuracy / len(val_loader)

        return val_loss, val_accuracy

class RoomTrainingPipeline:

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            device
    ):

        self.logger = get_logger(__name__)

        self.epoch_trainer = SingleEpochTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        self.validator = ValidationEvaluator(
            model=model,
            criterion=criterion,
            device=device
        )

    def train(
            self,
            train_loader,
            val_loader,
            epochs=1
    ):

        self.logger.info("Room classification training started")

        for epoch in range(epochs):

            train_loss, train_accuracy = (
                self.epoch_trainer.train_one_epoch(
                    train_loader
                )
            )

            val_loss, val_accuracy = (
                self.validator.validate(
                    val_loader
                )
            )

            self.logger.info(
                f"Epoch [{epoch + 1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

        self.logger.info("Room classification training completed")

class ModelSaver:

    def __init__(
            self,
            save_dir="data/pkl_files/deep_learning"
    ):

        self.project_root = Path(__file__).resolve().parents[2]
        self.save_dir = self.project_root / save_dir

        self.save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_model(
            self,
            model,
            class_names,
            file_name="room_classifier.pth"
    ):

        save_path = self.save_dir / file_name

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names
            },
            save_path
        )
        return save_path