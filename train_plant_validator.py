import random
import shutil
from pathlib import Path

import tensorflow as tf

# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

ALOE_DIR = BASE_DIR / "AloeVera_Plant"
UNSUPPORTED_DIR = (
    BASE_DIR / "plant_validation_data" / "unsupported_plants"
)

DATA_DIR = BASE_DIR / "validator_dataset"
MODEL_PATH = BASE_DIR / "plant_validator.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 15

SUPPORTED = "Supported"
UNSUPPORTED = "Unsupported"

random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------------------------------
# 2. Find images
# --------------------------------------------------

def get_images(folder):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if not folder.exists():
        return []

    return [
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
    ]


def split_images(images):
    images = list(images)
    random.shuffle(images)

    if len(images) < 2:
        raise ValueError(
            "Each class needs at least 2 images."
        )

    split_index = int(len(images) * 0.8)
    split_index = max(1, min(split_index, len(images) - 1))

    return images[:split_index], images[split_index:]


# --------------------------------------------------
# 3. Prepare dataset folders
# --------------------------------------------------

def reset_dataset():
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    for split in ["train", "val"]:
        for class_name in [SUPPORTED, UNSUPPORTED]:
            (DATA_DIR / split / class_name).mkdir(
                parents=True,
                exist_ok=True
            )


def copy_images(images, destination, prefix):
    for index, source in enumerate(images, start=1):
        target = destination / (
            f"{prefix}_{index}{source.suffix.lower()}"
        )
        shutil.copy2(source, target)

    print(f"{prefix}: copied {len(images)} images")


# --------------------------------------------------
# 4. Collect supported images: Aloe Vera ONLY
# --------------------------------------------------

aloe_images = get_images(ALOE_DIR / "train")

if not aloe_images:
    raise FileNotFoundError(
        "No Aloe Vera images found in AloeVera_Plant/train."
    )

print("Aloe Vera images found:", len(aloe_images))


# --------------------------------------------------
# 5. Collect unsupported images
# --------------------------------------------------

unsupported_images = get_images(UNSUPPORTED_DIR)

if not unsupported_images:
    raise FileNotFoundError(
        "No unsupported images found in "
        "plant_validation_data/unsupported_plants."
    )

print("Unsupported images found:", len(unsupported_images))

# Split each class separately
aloe_train, aloe_val = split_images(aloe_images)
unsupported_train, unsupported_val = split_images(
    unsupported_images
)

# --------------------------------------------------
# 6. Create training and validation folders
# --------------------------------------------------

reset_dataset()

copy_images(
    aloe_train,
    DATA_DIR / "train" / SUPPORTED,
    "aloe_train"
)

copy_images(
    aloe_val,
    DATA_DIR / "val" / SUPPORTED,
    "aloe_val"
)

copy_images(
    unsupported_train,
    DATA_DIR / "train" / UNSUPPORTED,
    "unsupported_train"
)

copy_images(
    unsupported_val,
    DATA_DIR / "val" / UNSUPPORTED,
    "unsupported_val"
)

print("\nDataset preparation completed.")


# --------------------------------------------------
# 7. Load datasets
# --------------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=[SUPPORTED, UNSUPPORTED],
    seed=SEED
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
# Supported = 0, Unsupported = 1

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# --------------------------------------------------
# 8. Build the validator model
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),

    # Input images are converted from 0-255 to 0-1.
    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),

    tf.keras.layers.Conv2D(
        32, 3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64, 3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        128, 3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        128, activation="relu"
    ),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        1, activation="sigmoid"
    )
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# --------------------------------------------------
# 9. Train the validator
# --------------------------------------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
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


# --------------------------------------------------
# 10. Evaluate and save
# --------------------------------------------------

loss, accuracy = model.evaluate(val_ds, verbose=0)

print("\nValidation loss:", round(float(loss), 4))
print("Validation accuracy:", round(float(accuracy) * 100, 2), "%")

model.save(MODEL_PATH)

print("\nValidator saved at:", MODEL_PATH)
print("Training completed.")/.nm,