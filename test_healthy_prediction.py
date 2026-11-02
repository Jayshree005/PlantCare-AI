from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "aloe_mobilenetv2_finetuned.keras"
HEALTHY_DIR = BASE_DIR / "AloeVera_Plant" / "test" / "Aloe_Healthy"

CLASS_NAMES = [
    "Aloe_Anthracnose",
    "Aloe_Healthy",
    "Aloe_Leafspot",
    "Aloe_Rust",
    "Aloe_Sunburn",
]

IMAGE_SIZE = (224, 224)
NUMBER_OF_IMAGES = 10

model = tf.keras.models.load_model(MODEL_PATH)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

healthy_images = sorted([
    path for path in HEALTHY_DIR.iterdir()
    if path.suffix.lower() in image_extensions
])

if not healthy_images:
    print("No healthy test images found. Check the dataset folder path.")
else:
    test_images = healthy_images[:NUMBER_OF_IMAGES]
    correct_predictions = 0

    print(f"\nTesting {len(test_images)} healthy Aloe Vera images...\n")

    for image_path in test_images:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMAGE_SIZE)

        image_array = np.array(img, dtype=np.float32)

        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        if predicted_class == "Aloe_Healthy":
            correct_predictions += 1

        print(f"Image: {image_path.name}")
        print(f"Predicted: {predicted_class}")
        print(f"Score: {confidence:.2f}%")

        print("All class probabilities:")
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"{class_name}: {predictions[i] * 100:.2f}%")
            
        print("-" * 40)

    accuracy = (correct_predictions / len(test_images)) * 100

    print("\n========== FINAL RESULT ==========")
    print(f"Healthy images tested: {len(test_images)}")
    print(f"Correctly predicted healthy: {correct_predictions}")
    print(f"Incorrectly classified: {len(test_images) - correct_predictions}")
    print(f"Healthy classification accuracy: {accuracy:.2f}%")