import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

TEST_DIR = "AloeVera_Plant/test"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# Load test dataset
test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = test_ds.class_names
print("\nDisease classes:", class_names)

# Load trained model
model = keras.models.load_model("aloe_cnn_best.keras")

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Actual labels
y_true = np.concatenate([
    labels.numpy() for images, labels in test_ds
])

# Predicted labels
predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
