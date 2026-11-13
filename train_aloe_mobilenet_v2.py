
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

# Extracted dataset - original dataset untouched
DATASET_DIR = Path(
    r"C:\Users\KANIKA\Downloads\plant disease (2)\plant disease\plant disease\AloeVera_Plant"
)

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

# New model filename - existing models will not be overwritten
MODEL_PATH = BASE_DIR / "aloe_mobilenetv2_extracted_v1.keras"

if MODEL_PATH.exists():
    raise FileExistsError(
        f"Model already exists: {MODEL_PATH}. "
        "Choose a new output filename before training."
    )

# Check dataset folders
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

tf.keras.utils.set_random_seed(42)

# Load datasets
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    seed=42
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# Verify class mapping
class_names = train_ds.class_names
num_classes = len(class_names)

if val_ds.class_names != class_names:
    raise ValueError("Training and validation class mapping differs.")

if test_ds.class_names != class_names:
    raise ValueError("Training and test class mapping differs.")

print("\nDisease Classes:", class_names)
print("Number of Classes:", num_classes)

# Calculate class weights
train_labels = np.concatenate([
    labels.numpy() for images, labels in train_ds
])

class_counts = np.bincount(train_labels, minlength=num_classes)

weights = len(train_labels) / (
    num_classes * np.maximum(class_counts, 1)
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(weights)
}

print("Class Weights:", class_weights)

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10)
])

# MobileNetV2 CNN
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Training callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )
]

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("\nStarting extracted Aloe dataset training...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# Load best saved model
best_model = keras.models.load_model(MODEL_PATH)

# Test evaluation
test_loss, test_accuracy = best_model.evaluate(test_ds)

print("\nFinal Test Results")
print("Test Loss:", test_loss)
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

y_true = np.concatenate([
    labels.numpy() for images, labels in test_ds
])

predictions = best_model.predict(test_ds, verbose=1)
y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report:")

cm = np.zeros((num_classes, num_classes), dtype=int)
np.add.at(cm, (y_true, y_pred), 1)

for i, name in enumerate(class_names):
    tp = cm[i, i]
    precision = tp / max(cm[:, i].sum(), 1)
    recall = tp / max(cm[i, :].sum(), 1)

    f1 = (
        2 * precision * recall /
        max(precision + recall, 1e-12)
    )

    support = cm[i, :].sum()

    print(
        f"{name}: "
        f"Precision={precision:.4f}, "
        f"Recall={recall:.4f}, "
        f"F1-score={f1:.4f}, "
        f"Support={support}"
    )

print("\nConfusion Matrix:")
print(cm)

print("\nModel saved at:", MODEL_PATH)