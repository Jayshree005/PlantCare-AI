import mimetypes
import os
import subprocess
import sys
import tempfile
import atexit
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
SAMPLE_IMAGE_DIR = (
    PROJECT_ROOT / "AloeVera_Plant" / "test" / "Aloe_Healthy"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def is_within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


def find_sample_image():
    if not SAMPLE_IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Aloe Healthy test-image folder was not found: {SAMPLE_IMAGE_DIR}"
        )

    candidates = sorted(
        (
            path
            for path in SAMPLE_IMAGE_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No supported image files were found in {SAMPLE_IMAGE_DIR}"
        )
    return candidates[0]


def child_test():
    temp_root = Path(os.environ["PLANTCARE_DIAG_TEMP_ROOT"]).resolve()
    project_root = Path(os.environ["PLANTCARE_DIAG_PROJECT_ROOT"]).resolve()
    sample_image = Path(os.environ["PLANTCARE_DIAG_SAMPLE_IMAGE"]).resolve()
    temp_db = temp_root / "isolated.sqlite3"
    temp_media_root = temp_root / "media"

    require(not is_within(temp_root, project_root),
            "The temporary directory must be outside the project directory.")
    require(is_within(sample_image, project_root),
            "The selected source image must be inside the project directory.")
    require(sample_image.is_file(), "The selected source image is missing.")
    require(is_within(temp_db, temp_root),
            "The temporary database path escaped the temporary directory.")
    require(is_within(temp_media_root, temp_root),
            "The temporary media path escaped the temporary directory.")

    sys.path.insert(0, str(BACKEND_DIR))
    os.environ["DJANGO_SETTINGS_MODULE"] = "plantcare_isolated_settings"

    import django

    django.setup()

    from django.conf import settings
    from django.core.files.uploadedfile import SimpleUploadedFile
    from django.core.management import call_command
    from django.core.files.storage import default_storage
    from django.db import connections, router
    from django.test import Client
    from detection.models import PredictionHistory

    atexit.register(connections.close_all)

    configured_db = Path(settings.DATABASES["default"]["NAME"]).resolve()
    configured_media = Path(settings.MEDIA_ROOT).resolve()
    require(configured_db == temp_db,
            f"Django database is not the temporary database: {configured_db}")
    require(is_within(configured_db, temp_root),
            "Django database is outside the temporary directory.")
    require(configured_media == temp_media_root,
            f"Django MEDIA_ROOT is not temporary: {configured_media}")
    require(is_within(configured_media, temp_root),
            "Django MEDIA_ROOT is outside the temporary directory.")
    require(not settings.DATABASE_ROUTERS,
            "Unexpected database routers could redirect database writes.")
    require(not router.routers,
            "Unexpected active database routers could redirect database writes.")
    require(
        Path(default_storage.location).resolve() == temp_media_root,
        "Django's default file storage is not using the temporary MEDIA_ROOT.",
    )
    require(
        settings.STORAGES["default"]["BACKEND"]
        == "django.core.files.storage.FileSystemStorage",
        "The configured default storage is not the expected filesystem storage.",
    )

    call_command("migrate", interactive=False, verbosity=0)

    history_db_alias = PredictionHistory.objects.db
    require(history_db_alias == "default",
            f"PredictionHistory selected unexpected database: {history_db_alias}")
    configured_history_db = Path(
        connections[history_db_alias].settings_dict["NAME"]
    ).resolve()
    require(configured_history_db == temp_db,
            "PredictionHistory is not using the temporary SQLite database.")

    content_type = mimetypes.guess_type(sample_image.name)[0] or "image/jpeg"
    with sample_image.open("rb") as image_file:
        uploaded_image = SimpleUploadedFile(
            name=sample_image.name,
            content=image_file.read(),
            content_type=content_type,
        )

    response = Client().post(
        "/api/upload/",
        {"plant": "aloe", "image": uploaded_image},
    )
    try:
        response_data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Upload endpoint returned non-JSON content: {response.content!r}"
        ) from exc

    print("Child process:", os.getpid())
    print("HTTP status:", response.status_code)
    print("JSON response:", response_data)
    require(response.status_code == 200,
            f"Expected HTTP 200, received {response.status_code}.")
    require(response_data.get("prediction_status") == "success",
            "The endpoint did not report a successful prediction.")
    require(response_data.get("plant") == "Aloe Vera",
            "The endpoint response did not identify Aloe Vera.")

    prediction = response_data.get("prediction")
    expected_classes = {
        "Aloe_Anthracnose",
        "Aloe_Healthy",
        "Aloe_Leafspot",
        "Aloe_Rust",
        "Aloe_Sunburn",
    }
    require(isinstance(prediction, str) and prediction in expected_classes,
            f"Unexpected Aloe prediction: {prediction!r}")

    confidence = response_data.get("confidence")
    require(
        isinstance(confidence, (int, float)) and 0 <= confidence <= 100,
        f"Invalid confidence value: {confidence!r}",
    )

    history_records = list(PredictionHistory.objects.all())
    require(
        len(history_records) == 1,
        f"Expected exactly one history record, found {len(history_records)}.",
    )
    record = history_records[0]
    require(record.plant == "aloe", "History record plant was not 'aloe'.")
    require(record.predicted_disease == prediction,
            "History prediction does not match the endpoint response.")
    require(round(record.confidence, 2) == round(confidence, 2),
            "History confidence does not match the endpoint response.")

    saved_image_path = Path(record.image.path).resolve()
    require(is_within(saved_image_path, temp_media_root),
            f"Uploaded image was saved outside temporary media: {saved_image_path}")
    require(saved_image_path.is_file(),
            f"Uploaded image was not found in temporary media: {saved_image_path}")

    current_db = Path(
        connections[history_db_alias].settings_dict["NAME"]
    ).resolve()
    require(current_db == temp_db,
            "The history database connection changed during the test.")
    require(is_within(saved_image_path, temp_root),
            "The uploaded file was saved outside the temporary directory.")

    print("Predicted disease:", prediction)
    print("Confidence:", confidence)
    print("History records:", len(history_records))
    print("Saved image:", saved_image_path)
    print("Temporary database:", configured_history_db)
    print("Isolated upload diagnostic passed.")
    connections.close_all()


def main():
    sample_image = find_sample_image()
    with tempfile.TemporaryDirectory(prefix="plantcare_upload_isolated_") as temp_dir:
        temp_root = Path(temp_dir).resolve()
        require(not is_within(temp_root, PROJECT_ROOT),
                "The temporary directory must be outside the project directory.")
        settings_module = temp_root / "plantcare_isolated_settings.py"
        settings_module.write_text(
            """
import os
from pathlib import Path

from config.settings import *

_TEMP_ROOT = Path(os.environ["PLANTCARE_DIAG_TEMP_ROOT"]).resolve()
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(_TEMP_ROOT / "isolated.sqlite3"),
        "OPTIONS": {"timeout": 30},
    }
}
MEDIA_ROOT = _TEMP_ROOT / "media"
STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
    },
}
DATABASE_ROUTERS = []
""".lstrip(),
            encoding="utf-8",
        )

        child_env = os.environ.copy()
        child_env["PLANTCARE_DIAG_TEMP_ROOT"] = str(temp_root)
        child_env["PLANTCARE_DIAG_PROJECT_ROOT"] = str(PROJECT_ROOT)
        child_env["PLANTCARE_DIAG_SAMPLE_IMAGE"] = str(sample_image)
        child_env["DJANGO_SETTINGS_MODULE"] = "plantcare_isolated_settings"
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        python_path = [
            str(temp_root),
            str(BACKEND_DIR),
        ]
        if child_env.get("PYTHONPATH"):
            python_path.append(child_env["PYTHONPATH"])
        child_env["PYTHONPATH"] = os.pathsep.join(python_path)

        print("Parent process:", os.getpid())
        print("Aloe Healthy test image:", sample_image)
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--child"],
            cwd=str(PROJECT_ROOT),
            env=child_env,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child_test()
    else:
        main()
