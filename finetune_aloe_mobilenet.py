import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Dataset paths
train_dir = "AloeVera_Plant/train"
val_dir = "AloeVera_Plant/val"
test_dir = "AloeVera_Plant/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names

print("\nClasses:", class_names)

# Calculate class weights
labels = np.concatenate([
    labels.numpy() for _, labels in train_ds
])

classes = np.unique(labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=labels
)

class_weights = dict(zip(classes, weights))

print("\nClass weights:", class_weights)

# Load previously trained model
model = load_model("aloe_mobilenetv2.keras")

print("\nOriginal model loaded successfully!")

# Find MobileNetV2 base model
base_model = model.get_layer("mobilenetv2_1.00_224")

# Unfreeze the base model
base_model.trainable = True

# Fine-tune only the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

print("\nTrainable layers:", sum(
    layer.trainable for layer in base_model.layers
))

# Compile with a low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Save best fine-tuned model separately
checkpoint = ModelCheckpoint(
    "aloe_mobilenetv2_finetuned.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=0.000001,
    verbose=1
)

# Fine-tune model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# Load best saved model
best_model = load_model("aloe_mobilenetv2_finetuned.keras")

# Evaluate on test dataset
print("\nEvaluating fine-tuned model...")

test_loss, test_accuracy = best_model.evaluate(test_ds)

print("\nTest Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

# Generate predictions
y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

predictions = best_model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))

print("\nFine-tuning completed!")