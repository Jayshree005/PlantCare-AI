
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Tomato_Plant"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

MODEL_PATH = BASE_DIR / "tomato_mobilenetv2.keras"
CLASS_NAMES_PATH = BASE_DIR / "tomato_class_names.json"

tf.keras.utils.set_random_seed(SEED)


# ---------------- VERIFY DATASET ----------------

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder missing: {folder}")

    class_folders = sorted(
        path.name for path in folder.iterdir()
        if path.is_dir()
    )

    if len(class_folders) != 10:
        raise ValueError(
            f"Expected 10 classes in {folder}, found {len(class_folders)}"
        )

train_classes = sorted(
    path.name for path in TRAIN_DIR.iterdir()
    if path.is_dir()
)

val_classes = sorted(
    path.name for path in VAL_DIR.iterdir()
    if path.is_dir()
)

test_classes = sorted(
    path.name for path in TEST_DIR.iterdir()
    if path.is_dir()
)

if not (train_classes == val_classes == test_classes):
    raise ValueError("Class names differ between dataset splits.")

print("Verified Tomato classes:", train_classes)


# ---------------- LOAD DATASETS ----------------

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=train_classes,
    shuffle=True,
    seed=SEED
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=train_classes,
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=train_classes,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

# Save exact class order for API prediction mapping.
with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

print("Number of classes:", num_classes)


# ---------------- CLASS WEIGHTS ----------------

train_labels = np.concatenate([
    labels.numpy() for images, labels in train_ds
])

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_labels
)

class_weights = dict(enumerate(weights))

print("Class weights:", class_weights)


# ---------------- DATA AUGMENTATION ----------------

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10)
])


# ---------------- MOBILENETV2 BASE MODEL ----------------

base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Initially freeze pretrained layers.
base_model.trainable = False


# ---------------- BUILD CNN MODEL ----------------

inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = layers.Rescaling(
    1.0 / 127.5,
    offset=-1
)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)


# ---------------- COMPILE ----------------

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ---------------- CALLBACKS ----------------

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


# ---------------- OPTIMIZE DATA PIPELINE ----------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ---------------- TRAIN ----------------

print("\nStarting Tomato MobileNetV2 training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# ---------------- EVALUATE BEST MODEL ----------------

print("\nLoading best saved Tomato model...")

best_model = keras.models.load_model(MODEL_PATH)

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

print("\nFinal Tomato Test Results")
print("Test Loss:", round(float(test_loss), 4))
print("Test Accuracy:", round(float(test_accuracy) * 100, 2), "%")


# ---------------- CLASSIFICATION REPORT ----------------

y_true = np.concatenate([
    labels.numpy() for images, labels in test_ds
])

predictions = best_model.predict(test_ds, verbose=1)

y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report:")

print(classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=class_names,
    zero_division=0
))


# ---------------- CONFUSION MATRIX ----------------

print("\nConfusion Matrix:")

print(confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(num_classes)
))


print("\nTomato model saved at:", MODEL_PATH)
print("Class mapping saved at:", CLASS_NAMES_PATH)
print("\nTomato CNN training completed.")
