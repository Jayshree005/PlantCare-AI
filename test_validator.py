import tensorflow as tf
import numpy as np
from pathlib import Path

MODEL_PATH = Path("plant_validator_v3.keras")

model = tf.keras.models.load_model(MODEL_PATH)

def test_image(image_path):
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    score = float(model.predict(image_array, verbose=0)[0][0])

    if score < 0.5:
        result = "Aloe Vera"
    else:
        result = "Non-Aloe Plant"

    print("\nImage:", image_path)
    print("Validator Score:", round(score, 4))
    print("Prediction:", result)


image_path = input("Enter image path: ").strip().strip('"')

if Path(image_path).exists():
    test_image(image_path)
else:
    print("Image not found!")