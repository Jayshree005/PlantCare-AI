
from .models import PredictionHistory
from pathlib import Path
import numpy as np
import tensorflow as tf

import json
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from PIL import Image, UnidentifiedImageError

from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status, generics

from .serializers import PredictionHistorySerializer


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "aloe_mobilenetv2_finetuned.keras"
VALIDATOR_MODEL_PATH = BASE_DIR / "plant_validator.keras"

# PlantCare AI supports Aloe Vera ONLY.
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
    print("PlantCare AI Aloe Vera CNN model loaded successfully!")
else:
    print(f"CNN model file not found: {MODEL_PATH}")


if VALIDATOR_MODEL_PATH.exists():
    validator_model = tf.keras.models.load_model(VALIDATOR_MODEL_PATH)
    print("Aloe Vera Plant Validator model loaded successfully!")
else:
    print(f"Validator model file not found: {VALIDATOR_MODEL_PATH}")


def is_valid_aloe_leaf(img):

    if validator_model is None:
        return None

    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)

    image_array = np.array(img, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = validator_model.predict(
        image_array,
        verbose=0
    )[0][0]

    print("Validator raw output:", prediction)
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

    if (
        not image_file.content_type
        or not image_file.content_type.startswith("image/")
    ):
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
                    "error": (
                        "This does not appear to be an Aloe Vera leaf. "
                        "Please upload an Aloe Vera image only."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        img = img.resize(IMAGE_SIZE)

        image_array = np.array(img, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

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
            {
                "error": (
                    "Could not read the uploaded image. "
                    "Please try another image."
                )
            },
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


# =====================================
# ALOE VERA WEATHER RISK CALCULATION
# =====================================

def calculate_aloe_weather_risk(
    disease,
    temperature,
    humidity,
    rain_probability,
    uv_index
):

    fungal_related = [
        "Aloe_Anthracnose",
        "Aloe_Leafspot",
        "Aloe_Rust"
    ]

    score = 0
    reasons = []
    recommendations = []

    if disease in fungal_related:

        if humidity is not None:
            if humidity >= 80:
                score += 2
                reasons.append(
                    "Very high humidity may favor leaf-wetness problems."
                )
            elif humidity >= 70:
                score += 1
                reasons.append(
                    "High humidity may increase favorable conditions."
                )

        if rain_probability is not None:
            if rain_probability >= 60:
                score += 2
                reasons.append(
                    "High rainfall probability may increase moisture exposure."
                )
            elif rain_probability >= 40:
                score += 1
                reasons.append(
                    "Rain is possible, so monitor leaf moisture."
                )

        if temperature is not None and 18 <= temperature <= 32:
            score += 1
            reasons.append(
                "Temperature may be favorable for some fungal diseases."
            )

        recommendations.extend([
            "Keep Aloe Vera leaves dry.",
            "Avoid overhead watering.",
            "Provide good airflow around the plant.",
            "Inspect leaves regularly for new spots."
        ])

    elif disease == "Aloe_Sunburn":

        if temperature is not None:
            if temperature >= 38:
                score += 2
                reasons.append(
                    "Very high temperature may increase heat stress."
                )
            elif temperature >= 35:
                score += 1
                reasons.append(
                    "High temperature may increase heat stress."
                )

        if uv_index is not None:
            if uv_index >= 8:
                score += 2
                reasons.append(
                    "Very high UV exposure may increase sunburn risk."
                )
            elif uv_index >= 6:
                score += 1
                reasons.append(
                    "High UV exposure may stress exposed leaves."
                )

        recommendations.extend([
            "Protect the plant from intense afternoon sunlight.",
            "Avoid sudden changes in sunlight exposure.",
            "Monitor leaves for bleaching or scorched patches."
        ])

    elif disease == "Aloe_Healthy":

        if temperature is not None and (
            temperature >= 35 or temperature <= 10
        ):
            score += 1
            reasons.append(
                "Temperature may cause environmental stress."
            )

        if humidity is not None and humidity >= 80:
            score += 1
            reasons.append(
                "High humidity requires careful watering and airflow."
            )

        recommendations.extend([
            "Continue suitable watering and provide proper drainage.",
            "Monitor leaves for new symptoms.",
            "Avoid sudden changes in sunlight."
        ])

    if score >= 4:
        risk_level = "High"
    elif score >= 2:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    if not reasons:
        reasons.append(
            "Current weather indicators do not show strong favorable conditions."
        )

    return {
        "risk_level": risk_level,
        "risk_score": score,
        "reasons": reasons,
        "recommendations": recommendations
    }


# =====================================
# WEATHER BY CITY
# =====================================

@api_view(["GET"])
def weather_by_city(request):

    city = request.GET.get("city", "").strip()

    if not city:
        return Response(
            {"error": "Please enter a city name."},
            status=status.HTTP_400_BAD_REQUEST
        )

    disease = request.GET.get(
        "disease",
        "Aloe_Healthy"
    ).strip()

    allowed_diseases = CLASS_NAMES

    if disease not in allowed_diseases:
        return Response(
            {
                "error": (
                    "Unsupported disease. "
                    "PlantCare AI currently supports Aloe Vera only."
                )
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    try:

        # =====================================
        # 1. FIND CITY COORDINATES
        # =====================================

        geo_url = "https://geocoding-api.open-meteo.com/v1/search?"

        geo_params = urlencode({
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        })

        with urlopen(geo_url + geo_params, timeout=15) as response:
            geo_data = json.loads(response.read().decode("utf-8"))

        if not geo_data.get("results"):
            return Response(
                {
                    "error": (
                        "City not found. "
                        "Please enter a valid city name."
                    )
                },
                status=status.HTTP_404_NOT_FOUND
            )

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # =====================================
        # 2. GET WEATHER DATA
        # =====================================

        weather_url = "https://api.open-meteo.com/v1/forecast?"

        weather_params = urlencode({
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "apparent_temperature,precipitation,"
                "weather_code,wind_speed_10m"
            ),
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,uv_index_max,"
                "relative_humidity_2m_mean"
            ),
            "forecast_days": 3,
            "timezone": "auto"
        })

        with urlopen(weather_url + weather_params, timeout=15) as response:
            weather_data = json.loads(response.read().decode("utf-8"))

        current = weather_data["current"]
        daily = weather_data["daily"]

        temperature = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]

        rain_probability = daily["precipitation_probability_max"][0]
        uv_index = daily["uv_index_max"][0]

        daily_humidity = daily["relative_humidity_2m_mean"]

        # =====================================
        # 3. CURRENT ALOE VERA CARE
        # =====================================

        recommendations = []

        if temperature >= 35:
            recommendations.append(
                "Protect Aloe Vera from extreme afternoon heat. "
                "Avoid sudden changes in sunlight."
            )

        elif temperature <= 10:
            recommendations.append(
                "Protect Aloe Vera from very cold temperatures."
            )

        else:
            recommendations.append(
                "Temperature is suitable for normal Aloe Vera care."
            )

        if humidity >= 75:
            recommendations.append(
                "Humidity is high. Avoid overwatering, keep leaves dry "
                "and maintain good airflow. Inspect regularly for new spots."
            )

        if rain_probability is not None and rain_probability >= 60:
            recommendations.append(
                "Rain is likely. Protect the plant from excess rainfall "
                "and check that the pot drains freely."
            )

        if (
            humidity >= 80
            and rain_probability is not None
            and rain_probability >= 60
        ):
            recommendations.append(
                "High humidity and likely rain can favor leaf-wetness "
                "problems. Keep leaves dry, improve airflow and monitor "
                "for new spots. Weather alone cannot diagnose disease."
            )

        if uv_index is not None and uv_index >= 6:
            recommendations.append(
                "UV levels are high. Avoid sudden exposure to intense sunlight."
            )

        # =====================================
        # 4. THREE-DAY DISEASE WEATHER RISK
        # =====================================

        daily_risk = []

        for index in range(len(daily["time"])):

            risk = calculate_aloe_weather_risk(
                disease=disease,
                temperature=daily["temperature_2m_max"][index],
                humidity=daily_humidity[index],
                rain_probability=daily["precipitation_probability_max"][index],
                uv_index=daily["uv_index_max"][index]
            )

            risk["date"] = daily["time"][index]
            risk["max_temperature"] = daily["temperature_2m_max"][index]
            risk["humidity"] = daily_humidity[index]
            risk["rain_probability"] = daily["precipitation_probability_max"][index]
            risk["uv_index"] = daily["uv_index_max"][index]

            daily_risk.append(risk)

        # =====================================
        # 5. RETURN WEATHER RESPONSE
        # =====================================

        return Response({

            "status": "success",

            "city": location["name"],

            "country": location.get("country", ""),

            "weather": {

                "temperature": temperature,

                "humidity": humidity,

                "feels_like": current["apparent_temperature"],

                "rainfall": current["precipitation"],

                "rain_probability": rain_probability,

                "wind_speed": current["wind_speed_10m"],

                "uv_index": uv_index,

                "weather_code": current["weather_code"]

            },

            "forecast": {

                "dates": daily["time"],

                "max_temperature": daily["temperature_2m_max"],

                "min_temperature": daily["temperature_2m_min"],

                "rain_probability": daily["precipitation_probability_max"],

                "humidity": daily_humidity,

                "uv_index": daily["uv_index_max"]

            },

            "aloe_care_recommendations": recommendations,

            "weather_disease_risk": {

                "plant": "Aloe Vera",

                "disease": disease,

                "method": "Rule-based weather risk estimation",

                "daily_risk": daily_risk

            }

        })

    except (HTTPError, URLError, TimeoutError):

        return Response(
            {"error": "Weather service is temporarily unavailable."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    except (ValueError, KeyError, IndexError, TypeError):

        return Response(
            {"error": "Unable to process weather data."},
            status=status.HTTP_502_BAD_GATEWAY
        )
