import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "PlantVillage")
OUTPUT_DIR = os.path.join(BASE_DIR, "plant_validation_data", "other_plants")

os.makedirs(OUTPUT_DIR, exist_ok=True)

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
random.seed(42)

total = 0

for folder in os.listdir(SOURCE_DIR):
    folder_path = os.path.join(SOURCE_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    images = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extensions):
                images.append(os.path.join(root, file))

    selected = random.sample(images, min(200, len(images)))

    for i, image in enumerate(selected, start=1):
        ext = os.path.splitext(image)[1].lower()
        new_name = f"{folder}_{i}{ext}"
        destination = os.path.join(OUTPUT_DIR, new_name)

        shutil.copy2(image, destination)
        total += 1

    print(folder, ":", len(selected), "images copied")

print("Total images copied:", total)
print("Output folder:", OUTPUT_DIR)