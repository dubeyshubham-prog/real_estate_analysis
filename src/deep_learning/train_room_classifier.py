from src.deep_learning.room_classifier import RoomClassifierTrainer
from src.deep_learning.trainer import RoomTrainingPipeline, ModelSaver

def train_room_classifier():
    trainer = RoomClassifierTrainer(
        subset_size=None,   # full dataset
        batch_size=32,
        image_size=224
    )

    trainer.prepare()

    pipeline = RoomTrainingPipeline(
        model=trainer.model,
        criterion=trainer.criterion,
        optimizer=trainer.optimizer,
        device=trainer.device
    )

    pipeline.train(
        train_loader=trainer.train_loader,
        val_loader=trainer.val_loader,
        epochs=5
    )

    saver = ModelSaver()

    save_path = saver.save_model(
        model=trainer.model,
        class_names=trainer.class_names,
        file_name="room_classifier.pth"
    )

    print(f"Final model saved at: {save_path}")


if __name__ == "__main__":
    train_room_classifier()