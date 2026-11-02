import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# Dataset path
DATASET_DIR = Path("AloeVera_Plant")

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5

# Load training dataset
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    seed=42
)

# Load validation dataset
val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# Load testing dataset
test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# Save class names
class_names = train_ds.class_names
print("\nDisease classes:", class_names)

# Improve data loading performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.cache().prefetch(
    buffer_size=AUTOTUNE
)

test_ds = test_ds.cache().prefetch(
    buffer_size=AUTOTUNE
)

# Build CNN model
model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(len(class_names), activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model structure
model.summary()

# Save best model during training
checkpoint = keras.callbacks.ModelCheckpoint(
    "aloe_cnn_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

# Train CNN
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# Evaluate on test dataset
test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Save final model
model.save("aloe_cnn_final.keras")

print("\nTraining complete!")
print("Best model: aloe_cnn_best.keras")
print("Final model: aloe_cnn_final.keras")