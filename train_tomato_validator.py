
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix


# =====================================
# 1. PATHS AND SETTINGS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "Tomato_Validator_Dataset"
MODEL_PATH = BASE_DIR / "tomato_validator.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

SUPPORTED = "Supported"
UNSUPPORTED = "Unsupported"

tf.random.set_seed(SEED)
np.random.seed(SEED)

if MODEL_PATH.exists():
    raise FileExistsError(
        f"{MODEL_PATH.name} already exists. "
        "Rename or back up the existing model before training."
    )


# =====================================
# 2. CHECK DATASET
# =====================================

for split in ["train", "val", "test"]:
    for label in [SUPPORTED, UNSUPPORTED]:
        folder = DATA_DIR / split / label

        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder missing: {folder}"
            )

print("Dataset folders verified.")


# =====================================
# 3. LOAD DATASETS
# =====================================

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

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=[SUPPORTED, UNSUPPORTED],
    shuffle=False
)

print("\nClass mapping:", train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# =====================================
# 4. BUILD CNN MODEL
# =====================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(224, 224, 3)),

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


# =====================================
# 5. COMPILE MODEL
# =====================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()


# =====================================
# 6. CLASS WEIGHTS
# =====================================

total = 3755 + 1986

class_weights = {
    0: total / (2 * 3755),
    1: total / (2 * 1986)
}

print("\nClass weights:", class_weights)


# =====================================
# 7. TRAINING CALLBACKS
# =====================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )
]


# =====================================
# 8. TRAIN MODEL
# =====================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)


# =====================================
# 9. LOAD BEST MODEL
# =====================================

best_model = tf.keras.models.load_model(
    MODEL_PATH
)


# =====================================
# 10. TEST EVALUATION
# =====================================

test_loss, test_accuracy, test_precision, test_recall = (
    best_model.evaluate(test_ds, verbose=1)
)

print("\n========== TEST RESULTS ==========")

print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")
print("Test Precision:", round(test_precision * 100, 2), "%")
print("Test Recall:", round(test_recall * 100, 2), "%")


# =====================================
# 11. CONFUSION MATRIX
# =====================================

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = best_model.predict(
        images,
        verbose=0
    ).ravel()

    y_true.extend(
        labels.numpy().astype(int).ravel()
    )

    y_pred.extend(
        (predictions >= 0.5).astype(int)
    )


print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(y_true, y_pred)
)


# =====================================
# 12. CLASSIFICATION REPORT
# =====================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Supported Tomato",
            "Unsupported Plant"
        ],
        zero_division=0
    )
)


print("\nTomato Validator training completed!")
print("Model saved at:", MODEL_PATH)
print("Output 0 = Supported Tomato")
print("Output 1 = Unsupported Plant")
