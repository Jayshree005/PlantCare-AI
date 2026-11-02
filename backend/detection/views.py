from .models import PredictionHistory
from pathlib import Path
import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError

from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status, generics

from .serializers import PredictionHistorySerializer


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "aloe_mobilenetv2_finetuned.keras"
VALIDATOR_MODEL_PATH = BASE_DIR / "plant_validator.keras"

CLASS_NAMES = [
    "Aloe_Anthracnose",
    "Aloe_Healthy",
    "Aloe_Leafspot",
    "Aloe_Rust",
    "Aloe_Sunburn",
]

IMAGE_SIZE = (224, 224)

model = None
validator_model = None

if MODEL_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)
    print("PlantCare AI CNN model loaded successfully!")
else:
    print(f"CNN model file not found: {MODEL_PATH}")

if VALIDATOR_MODEL_PATH.exists():
    validator_model = tf.keras.models.load_model(VALIDATOR_MODEL_PATH)
    print("Plant Validator model loaded successfully!")
else:
    print(f"Validator model file not found: {VALIDATOR_MODEL_PATH}")


def is_valid_aloe_leaf(img):
    if validator_model is None:
        return None

    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)

    image_array = np.array(img, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = validator_model.predict(image_array, verbose=0)[0][0]
    print("Validator raw output:", validator_model.predict(image_array, verbose=0))

    print("Validator score:", float(prediction))
    print("Validator input image size:", img.size)
    
    return float(prediction) < 0.5

    


@api_view(["GET"])
def api_home(request):
    return Response({
        "message": "PlantCare AI API is working!",
        "status": "success",
        "cnn_model_loaded": model is not None,
        "validator_model_loaded": validator_model is not None,
        "supported_plant": "Aloe Vera",
        "supported_classes": CLASS_NAMES,
    })


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload_leaf(request):
    image_file = request.FILES.get("image")

    if image_file is None:
        return Response(
            {"error": "Please upload an image using the 'image' field."},
            status=status.HTTP_400_BAD_REQUEST
        )

    if not image_file.content_type or not image_file.content_type.startswith("image/"):
        return Response(
            {"error": "Please upload a valid image file."},
            status=status.HTTP_400_BAD_REQUEST
        )

    if model is None:
        return Response(
            {"error": "CNN model is not available."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    if validator_model is None:
        return Response(
            {"error": "Plant validator model is not available."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    try:
        img = Image.open(image_file).convert("RGB")
        
        print("DEBUG: Checking Aloe Vera image")

        is_aloe = is_valid_aloe_leaf(img)
        
        print("DEBUG: Aloe validation result:", is_aloe)

        if is_aloe is None:
            return Response(
                {"error": "Plant validation is unavailable."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        if not is_aloe:
            return Response(
                {
                    "prediction_status": "rejected",
                    "error": "This does not appear to be an Aloe Vera leaf. Please upload an Aloe Vera image."
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        img = img.resize(IMAGE_SIZE)

        image_array = np.array(img, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        print("Prediction:", predicted_class)
        print("Confidence:", round(confidence, 2), "%")

        PredictionHistory.objects.create(
            image=image_file,
            predicted_disease=predicted_class,
            confidence=round(confidence, 2),
        )

        return Response({
            "message": "Image analyzed successfully.",
            "filename": image_file.name,
            "prediction_status": "success",
            "plant": "Aloe Vera",
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
        }, status=status.HTTP_200_OK)

    except (UnidentifiedImageError, OSError, ValueError):
        return Response(
            {"error": "Could not read the uploaded image. Please try another image."},
            status=status.HTTP_400_BAD_REQUEST
        )

    except Exception as e:
        return Response(
            {"error": f"Prediction failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class PredictionHistoryListView(generics.ListAPIView):
    queryset = PredictionHistory.objects.all().order_by("-prediction_date")
    serializer_class = PredictionHistorySerializer 