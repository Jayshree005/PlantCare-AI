
from pathlib import Path
import random
import shutil

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "PlantVillage"
OUTPUT_DIR = BASE_DIR / "Tomato_Plant"

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

TOMATO_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
]

random.seed(RANDOM_SEED)

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists. "
        "Rename or inspect it before preparing the dataset."
    )

# Verify all source folders before copying anything.
for class_name in TOMATO_CLASSES:
    folder = SOURCE_DIR / class_name

    if not folder.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {folder}")

    images = [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(images) < 3:
        raise ValueError(f"Not enough images in {class_name}")

print("All 10 Tomato class folders verified.")

try:
    for split in ["train", "val", "test"]:
        for class_name in TOMATO_CLASSES:
            (OUTPUT_DIR / split / class_name).mkdir(
                parents=True,
                exist_ok=True
            )

    for class_name in TOMATO_CLASSES:
        source_folder = SOURCE_DIR / class_name

        images = [
            path for path in source_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        images.sort()
        random.shuffle(images)

        total = len(images)

        train_end = int(total * 0.70)
        val_end = train_end + int(total * 0.15)

        split_data = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split, split_images in split_data.items():
            destination = OUTPUT_DIR / split / class_name

            for image_path in split_images:
                shutil.copy2(
                    image_path,
                    destination / image_path.name
                )

        print(
            f"{class_name}: "
            f"Train={len(split_data['train'])}, "
            f"Val={len(split_data['val'])}, "
            f"Test={len(split_data['test'])}"
        )

except Exception:
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    raise

print("\nTomato dataset preparation completed.")
print("Dataset location:", OUTPUT_DIR)
