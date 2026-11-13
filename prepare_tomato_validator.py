
import random
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

TOMATO_DIR = BASE_DIR / "Tomato_Plant"
PLANTVILLAGE_DIR = BASE_DIR / "PlantVillage"
FLOWERS_DIR = BASE_DIR / "plant_validation_data" / "unsupported_plants"

OUTPUT_DIR = BASE_DIR / "Tomato_Validator_Dataset"

SEED = 42
random.seed(SEED)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Do not overwrite an existing dataset
if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists. "
        "Rename or inspect it before running this script again."
    )


def get_images(folder):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    return [
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    ]


# 1. Collect unsupported plant images
negative_images = []

negative_folders = [
    PLANTVILLAGE_DIR / "Pepper__bell___Bacterial_spot",
    PLANTVILLAGE_DIR / "Pepper__bell___healthy",
    PLANTVILLAGE_DIR / "Potato___Early_blight",
    PLANTVILLAGE_DIR / "Potato___healthy",
    PLANTVILLAGE_DIR / "Potato___Late_blight",
]

for folder in negative_folders:
    negative_images.extend(get_images(folder))

for folder in FLOWERS_DIR.iterdir():
    if folder.is_dir():
        negative_images.extend(get_images(folder))

# Remove duplicate paths
negative_images = list(set(negative_images))
random.shuffle(negative_images)

# 2. Split unsupported images: 70/15/15
total = len(negative_images)

train_end = int(total * 0.70)
val_end = train_end + int(total * 0.15)

negative_splits = {
    "train": negative_images[:train_end],
    "val": negative_images[train_end:val_end],
    "test": negative_images[val_end:],
}

# 3. Create output folders
for split in ["train", "val", "test"]:
    for label in ["Supported", "Unsupported"]:
        (OUTPUT_DIR / split / label).mkdir(
            parents=True,
            exist_ok=True
        )


def copy_images(images, destination, prefix):
    for index, source in enumerate(images):
        target = destination / (
            f"{prefix}_{index}{source.suffix.lower()}"
        )
        shutil.copy2(source, target)


# 4. Copy existing Tomato splits as Supported
for split in ["train", "val", "test"]:
    tomato_images = get_images(TOMATO_DIR / split)

    copy_images(
        tomato_images,
        OUTPUT_DIR / split / "Supported",
        "tomato"
    )

    copy_images(
        negative_splits[split],
        OUTPUT_DIR / split / "Unsupported",
        "other"
    )

# 5. Print final counts
print("\nTOMATO VALIDATOR DATASET CREATED")

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}")

    for label in ["Supported", "Unsupported"]:
        count = len(get_images(OUTPUT_DIR / split / label))
        print(f"{label}: {count}")

print("\nDataset location:", OUTPUT_DIR)
print("Original datasets were not modified.")
