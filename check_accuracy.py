import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Dataset folders
TEST_DIR = BASE_DIR / "AloeVera_Plant" / "test"

# Model path
MODEL_PATH = BASE_DIR / "aloe_cnn_best.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

print("Loading test dataset...")

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("\nTest classes:", test_ds.class_names)

print("\nLoading trained model...")

model = keras.models.load_model(MODEL_PATH)

print("\nEvaluating model on test dataset...")

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print("\n========== TEST RESULTS ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("==================================")