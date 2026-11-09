import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


# 1. PATHS AND SETTINGS

BASE_DIR = Path(__file__).resolve().parent

ALOE_DIR = BASE_DIR / "AloeVera_Plant" / "train"
REAL_ALOE_DIR = BASE_DIR / "AloeVera_Plant" / "real_world_aloe"

UNSUPPORTED_DIR = BASE_DIR / "plant_validation_data" / "unsupported_plants"
OTHER_PLANTS_DIR = BASE_DIR / "plant_validation_data" / "other_plants"

DATA_DIR = BASE_DIR / "validator_dataset_v3"
MODEL_PATH = BASE_DIR / "plant_validator_v3.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 20

SUPPORTED = "Supported"
UNSUPPORTED = "Unsupported"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# 2. FIND IMAGES

def get_images(folder):
    if not folder.exists():
        return []

    return [
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    ]


aloe_images = get_images(ALOE_DIR)
real_aloe_images = get_images(REAL_ALOE_DIR)

unsupported_images = (
    get_images(UNSUPPORTED_DIR) +
    get_images(OTHER_PLANTS_DIR)
)

print("Dataset Aloe images:", len(aloe_images))
print("Real-world Aloe images:", len(real_aloe_images))
print("Unsupported images:", len(unsupported_images))

if not aloe_images or not unsupported_images:
    raise ValueError("One or both classes have no images.")


# 3. ADD REAL-WORLD ALOE IMAGES

aloe_images = list(set(aloe_images + real_aloe_images))

print("Total Supported Aloe images:", len(aloe_images))


# 4. BALANCE DATASET

random.shuffle(aloe_images)
random.shuffle(unsupported_images)

real_aloe_set = set(real_aloe_images)

original_aloe_images = [
    img for img in aloe_images
    if img not in real_aloe_set
]

samples_per_class = min(
    len(original_aloe_images),
    len(unsupported_images)
)

original_aloe_images = original_aloe_images[:samples_per_class]
unsupported_images = unsupported_images[:samples_per_class]


# 5. SPLIT DATASET

def split_images(images):
    split_index = int(len(images) * 0.8)
    return images[:split_index], images[split_index:]


aloe_train, aloe_val = split_images(original_aloe_images)

# Include all real-world Aloe images in training
aloe_train.extend(real_aloe_images)

random.shuffle(aloe_train)

unsupported_train, unsupported_val = split_images(unsupported_images)


print("\nDATASET SPLIT RESULTS")

print("Original Aloe training:", len(aloe_train) - len(real_aloe_images))
print("Real-world Aloe training:", len(real_aloe_images))
print("Total Aloe training:", len(aloe_train))
print("Aloe validation:", len(aloe_val))
print("Unsupported training:", len(unsupported_train))
print("Unsupported validation:", len(unsupported_val))


# 6. CREATE DATASET FOLDERS

if DATA_DIR.exists():
    shutil.rmtree(DATA_DIR)

for split in ["train", "val"]:
    for class_name in [SUPPORTED, UNSUPPORTED]:
        (DATA_DIR / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )


def copy_images(images, destination, prefix):
    for index, source in enumerate(images):
        target = destination / f"{prefix}_{index}{source.suffix.lower()}"
        shutil.copy2(source, target)


copy_images(aloe_train, DATA_DIR / "train" / SUPPORTED, "aloe")
copy_images(aloe_val, DATA_DIR / "val" / SUPPORTED, "aloe")

copy_images(
    unsupported_train,
    DATA_DIR / "train" / UNSUPPORTED,
    "other"
)

copy_images(
    unsupported_val,
    DATA_DIR / "val" / UNSUPPORTED,
    "other"
)

print("\nDataset preparation completed.")


# 7. LOAD DATASETS

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=[SUPPORTED, UNSUPPORTED],
    seed=SEED,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=[SUPPORTED, UNSUPPORTED],
    shuffle=False
)

print("Class mapping:", train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# 8. BUILD CNN MODEL

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),

    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(1, activation="sigmoid")
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()


# 9. TRAIN MODEL

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True
    )
]


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# 10. EVALUATE MODEL

best_model = tf.keras.models.load_model(MODEL_PATH)

loss, accuracy, precision, recall = best_model.evaluate(
    val_ds,
    verbose=1
)

print("\nVALIDATION RESULTS")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")


# 11. CONFUSION MATRIX

y_true = []
y_pred = []

for images, labels in val_ds:
    scores = best_model.predict(images, verbose=0).ravel()

    y_true.extend(labels.numpy().astype(int))
    y_pred.extend((scores >= 0.5).astype(int))


print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Aloe Vera", "Non-Aloe"]
    )
)

print("\nNew validator saved at:", MODEL_PATH)
print("Backup model remains safe.")