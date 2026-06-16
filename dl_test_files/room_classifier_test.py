from src.deep_learning.room_classifier import (
    RoomClassifierTrainer
)


from src.deep_learning.trainer import (
    RoomTrainingPipeline,
    ModelSaver
)
from torch.utils.data import Subset

def test_room_classifier_pipeline():

    trainer = RoomClassifierTrainer(
        subset_size=500
    )
    trainer.prepare()

    pipeline = RoomTrainingPipeline(
        model=trainer.model,
        criterion=trainer.criterion,
        optimizer=trainer.optimizer,
        device=trainer.device
    )

    print("\nPipeline Created Successfully")

    print("\nStarting Training Test...")

    pipeline.train(
        train_loader=trainer.train_loader,
        val_loader=trainer.val_loader,
        epochs=1
    )

    saver = ModelSaver()

    save_path = saver.save_model(
        model=trainer.model,
        class_names=trainer.class_names,
        file_name="room_classifier_test.pth"
    )
    print(f"\nModel saved at: {save_path}")

    return pipeline

if __name__ == "__main__":
    test_room_classifier_pipeline()