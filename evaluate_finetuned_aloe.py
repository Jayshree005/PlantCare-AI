from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Model and dataset paths
MODEL_PATH = BASE_DIR / "aloe_mobilenetv2_finetuned.keras"
TEST_DIR = BASE_DIR / "AloeVera_Plant" / "test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("\nClasses:")
print(class_names)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

print("\nFine-tuned model loaded successfully!")

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print("\n========== MODEL EVALUATION ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Actual labels
y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

# Predicted probabilities
predictions = model.predict(test_ds, verbose=1)

# Predicted class indices
y_pred = np.argmax(predictions, axis=1)

# Classification report
print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

# Confusion matrix
print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_true, y_pred))

# Correct and incorrect predictions
correct = np.sum(y_true == y_pred)
incorrect = np.sum(y_true != y_pred)

print("\n========== FINAL RESULT ==========")
print(f"Total test images: {len(y_true)}")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {incorrect}")
print(f"Accuracy: {correct / len(y_true) * 100:.2f}%")