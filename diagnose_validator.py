import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "plant_validator_v3.keras"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_first_image_for_each_class(root: Path):
    result = {}
    if not root.exists():
        return result

    for split_dir in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not split_dir.is_dir():
            continue

        for class_dir in sorted(split_dir.iterdir(), key=lambda p: p.name.lower()):
            if not class_dir.is_dir():
                continue

            files = sorted(
                [
                    p for p in class_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
                ],
                key=lambda p: p.name.lower(),
            )

            if files and class_dir.name not in result:
                result[class_dir.name] = files[0]

    return result


def predict_score(image_path: Path):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    score = float(model.predict(arr, verbose=0)[0][0])
    decision = "SUPPORTED ALOE" if score < 0.5 else "UNSUPPORTED PLANT"
    return score, decision


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Validator model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

print("Validator model:", MODEL_PATH.name)

aloe_classes = find_first_image_for_each_class(BASE_DIR / "AloeVera_Plant")
if aloe_classes:
    print("\nAloe images discovered:")
    for class_name, image_path in aloe_classes.items():
        score, decision = predict_score(image_path)
        print(
            f"- {class_name}: {image_path} | "
            f"score={score:.8f} | decision={decision}"
        )
else:
    print("No Aloe images found under AloeVera_Plant.")

tomato_path = None
tomato_root = BASE_DIR / "PlantVillage"
if tomato_root.exists():
    for folder in sorted(tomato_root.iterdir(), key=lambda p: p.name.lower()):
        if not folder.is_dir():
            continue

        files = sorted(
            [
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
            ],
            key=lambda p: p.name.lower(),
        )

        if files:
            tomato_path = files[0]
            break

if tomato_path:
    score, decision = predict_score(tomato_path)
    print(
        f"\nTomato sample: {tomato_path} | "
        f"score={score:.8f} | decision={decision}"
    )
else:
    print("\nNo Tomato image found in PlantVillage.")

print("\nInterpretation:")
print("This validator was trained with label logic: 0 = supported Aloe, 1 = unsupported plant.")
print("So score < 0.5 => supported Aloe; score >= 0.5 => unsupported plant.")
print("This command does not modify or retrain any model or dataset.")
