import os
import shutil
import sys
import tempfile
from pathlib import Path

import django
from django.core.management import call_command
from django.test import Client, override_settings
from django.core.files.uploadedfile import SimpleUploadedFile

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from detection.models import PredictionHistory


def find_first_image(root: Path, class_name: str):
    class_dir = root / class_name
    if not class_dir.exists():
        return None

    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    candidates = []

    for path in class_dir.iterdir():
        if path.is_file() and path.suffix.lower() in allowed_exts:
            candidates.append(path)

    if not candidates:
        return None

    return sorted(candidates, key=lambda p: p.name.lower())[0]


def main():
    project_root = BASE_DIR
    aloe_root = project_root / "AloeVera_Plant"
    sample_image = find_first_image(aloe_root / "test", "Aloe_Healthy")

    if sample_image is None:
        raise FileNotFoundError("No Aloe Healthy image found in AloeVera_Plant/test/Aloe_Healthy.")

    with tempfile.TemporaryDirectory(prefix="plantcare_upload_diag_") as temp_dir:
        temp_media_root = Path(temp_dir) / "media"
        temp_media_root.mkdir(parents=True, exist_ok=True)
        temp_db = Path(temp_dir) / "test_upload_diag.sqlite3"

        with override_settings(
            DATABASES={
                "default": {
                    "ENGINE": "django.db.backends.sqlite3",
                    "NAME": str(temp_db),
                }
            },
            MEDIA_ROOT=str(temp_media_root),
            DEFAULT_FILE_STORAGE="django.core.files.storage.FileSystemStorage",
        ):
            call_command("migrate", run_syncdb=True, verbosity=0)

            with open(sample_image, "rb") as image_file:
                uploaded = SimpleUploadedFile(
                    name=sample_image.name,
                    content=image_file.read(),
                    content_type="image/jpeg",
                )

                client = Client()
                response = client.post(
                    "/api/upload/",
                    {"plant": "aloe", "image": uploaded},
                    format="multipart",
                )

            print("HTTP status:", response.status_code)
            print("JSON response:", response.json())

            history_records = list(PredictionHistory.objects.all())
            print("History count:", len(history_records))

            if history_records:
                record = history_records[0]
                print("Predicted disease:", record.predicted_disease)
                print("Confidence:", record.confidence)
                print("Saved image path:", record.image.path)
                print("Image file exists:", Path(record.image.path).exists())

            print("Temp media root:", temp_media_root)
            print("Temp db path:", temp_db)
            print("Temporary database file exists:", temp_db.exists())

            if history_records:
                assert response.status_code == 200, "Expected HTTP 200 for valid Aloe upload."
                assert response.json().get("prediction_status") == "success", "Expected success response."
                assert record.predicted_disease in {
                    "Aloe_Anthracnose",
                    "Aloe_Healthy",
                    "Aloe_Leafspot",
                    "Aloe_Rust",
                    "Aloe_Sunburn",
                }, "Prediction disease should be one of the known Aloe classes."
                assert Path(record.image.path).exists(), "PredictionHistory image file missing from temp media root."
                assert len(history_records) == 1, "Expected exactly one temporary prediction-history row."

            print("Temporary upload diagnostic completed successfully.")


if __name__ == "__main__":
    main()
