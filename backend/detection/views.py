
from .models import PredictionHistory
from pathlib import Path
from datetime import datetime
import re
import numpy as np
import tensorflow as tf

import json
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from PIL import Image, UnidentifiedImageError

from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.models import User

from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status, generics

from .serializers import PredictionHistorySerializer
from django.views.decorators.csrf import ensure_csrf_cookie


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "aloe_mobilenetv2_extracted_v1.keras"
VALIDATOR_MODEL_PATH = BASE_DIR / "plant_validator_v3.keras"

# Tomato model paths
TOMATO_MODEL_PATH = BASE_DIR / "tomato_mobilenetv2.keras"
TOMATO_VALIDATOR_MODEL_PATH = BASE_DIR / "tomato_validator.keras"
TOMATO_CLASS_NAMES_PATH = BASE_DIR / "tomato_class_names.json"

# PlantCare AI supports Aloe Vera and Tomato.
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

tomato_model = None
tomato_validator_model = None
TOMATO_CLASS_NAMES = []

# Load Tomato class names
try:
    with open(TOMATO_CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        TOMATO_CLASS_NAMES = json.load(file)

    print("Tomato class names loaded successfully!")

except (OSError, json.JSONDecodeError) as e:
    print("Tomato class names could not be loaded:", e)


# Load Tomato CNN model
try:
    if TOMATO_MODEL_PATH.exists():
        tomato_model = tf.keras.models.load_model(TOMATO_MODEL_PATH)
        print("Tomato CNN model loaded successfully!")
    else:
        print("Tomato CNN model file not found:", TOMATO_MODEL_PATH)

except Exception as e:
    print("Tomato CNN loading failed:", e)


# Load Tomato Validator model
try:
    if TOMATO_VALIDATOR_MODEL_PATH.exists():
        tomato_validator_model = tf.keras.models.load_model(
            TOMATO_VALIDATOR_MODEL_PATH
        )
        print("Tomato Validator model loaded successfully!")
    else:
        print("Tomato Validator model file not found:", TOMATO_VALIDATOR_MODEL_PATH)

except Exception as e:
    print("Tomato Validator loading failed:", e)


if MODEL_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)
    print("PlantCare AI Aloe Vera CNN model loaded successfully!")

    # Temporary Aloe CNN diagnostic
    print("\n========== ALOE MODEL CHECK ==========")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Rescaling):
            print("Rescaling layer found:", layer.get_config())

    print("======================================\n")

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

    print(
        "Validator decision:",
        "SUPPORTED ALOE" if float(prediction) < 0.5
        else "UNSUPPORTED PLANT"
    )

    print("Validator model:", VALIDATOR_MODEL_PATH.name)

    return float(prediction) < 0.5


def is_valid_tomato_leaf(img):

    if tomato_validator_model is None:
        return None

    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)

    image_array = np.array(img, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = tomato_validator_model.predict(
        image_array,
        verbose=0
    )[0][0]

    print("Tomato Validator raw output:", prediction)

    # 0 = Supported Tomato, 1 = Unsupported Plant
    return float(prediction) < 0.5


@api_view(["POST"])
def signup_user(request):
    first_name = (request.data.get("name") or "").strip()
    email = (request.data.get("email") or "").strip().lower()
    password = request.data.get("password") or ""

    if not first_name or not email or not password:
        return Response(
            {"error": "Name, email and password are required."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if "@" not in email:
        return Response(
            {"error": "Please provide a valid email address."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if len(password) < 8:
        return Response(
            {"error": "Password must be at least 8 characters long."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if not re.search(r"[^A-Za-z0-9\s]", password):
        return Response(
            {"error": "Password must contain at least one special character."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if User.objects.filter(email__iexact=email).exists():
        return Response(
            {"error": "An account with this email already exists."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    user = User.objects.create_user(
        username=email,
        email=email,
        password=password,
        first_name=first_name,
    )

    return Response(
        {
            "message": "Account created successfully.",
            "user": {
                "id": user.id,
                "name": user.get_full_name() or user.username,
                "email": user.email,
            },
        },
        status=status.HTTP_201_CREATED,
    )


@api_view(["POST"])
def login_user(request):
    email = (request.data.get("email") or "").strip().lower()
    password = request.data.get("password") or ""

    if not email or not password:
        return Response(
            {"error": "Email and password are required."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    user = authenticate(request, username=email, password=password)
    if user is None:
        return Response(
            {"error": "Invalid email or password."},
            status=status.HTTP_401_UNAUTHORIZED,
        )

    auth_login(request, user)

    return Response(
        {
            "message": "Login successful.",
            "user": {
                "id": user.id,
                "name": user.get_full_name() or user.username,
                "email": user.email,
            },
        },
        status=status.HTTP_200_OK,
    )


@api_view(["POST"])
def logout_user(request):
    auth_logout(request)
    return Response({"message": "Logged out successfully."}, status=status.HTTP_200_OK)


@api_view(["GET"])
def current_user(request):
    if not request.user.is_authenticated:
        return Response({"error": "Not authenticated."}, status=status.HTTP_401_UNAUTHORIZED)

    return Response(
        {
            "id": request.user.id,
            "name": request.user.get_full_name() or request.user.username,
            "email": request.user.email,
        },
        status=status.HTTP_200_OK,
    )


@api_view(["GET"])
def api_home(request):

    return Response({
        "message": "PlantCare AI API is working!",
        "status": "success",

        # Aloe compatibility
        "cnn_model_loaded": model is not None,
        "validator_model_loaded": validator_model is not None,
        "supported_plant": "Aloe Vera",
        "supported_classes": CLASS_NAMES,

        # Both supported plants
        "plants": {
            "aloe": {
                "cnn_model_loaded": model is not None,
                "validator_model_loaded": validator_model is not None,
                "classes": CLASS_NAMES
            },
            "tomato": {
                "cnn_model_loaded": tomato_model is not None,
                "validator_model_loaded": tomato_validator_model is not None,
                "classes": TOMATO_CLASS_NAMES
            }
        }
    })


@api_view(["POST"])
@permission_classes([AllowAny])
@parser_classes([MultiPartParser, FormParser])
def upload_leaf(request):

    image_file = request.FILES.get("image")
    plant = request.data.get("plant", "aloe").strip().lower()

    if image_file is None:
        return Response(
            {"error": "Please upload an image using the 'image' field."},
            status=status.HTTP_400_BAD_REQUEST
        )

    if plant not in ["aloe", "tomato"]:
        return Response(
            {"error": "Unsupported plant. Select Aloe Vera or Tomato."},
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

    if plant == "aloe":
        selected_model = model
        selected_validator = validator_model
        selected_classes = CLASS_NAMES
        plant_name = "Aloe Vera"
        validator_function = is_valid_aloe_leaf

    else:
        selected_model = tomato_model
        selected_validator = tomato_validator_model
        selected_classes = TOMATO_CLASS_NAMES
        plant_name = "Tomato"
        validator_function = is_valid_tomato_leaf

    if selected_model is None or selected_validator is None:
        return Response(
            {"error": f"{plant_name} models are not available."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    if not selected_classes:
        return Response(
            {"error": f"{plant_name} class names are unavailable."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    try:

        img = Image.open(image_file).convert("RGB")

        print(f"Checking {plant_name} image")

        is_valid_plant = validator_function(img)

        if is_valid_plant is None:
            return Response(
                {"error": "Plant validation is unavailable."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        if not is_valid_plant:
            return Response(
                {
                    "prediction_status": "rejected",
                    "error": (
                        f"This does not appear to be a {plant_name} leaf. "
                        f"Please upload a {plant_name} image only."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        img = img.resize(IMAGE_SIZE)

        image_array = np.array(img, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = selected_model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(np.argmax(predictions))

        if predicted_index >= len(selected_classes):
            return Response(
                {"error": "Invalid prediction class mapping."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        predicted_class = selected_classes[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        print("\n========== ALL CLASS PROBABILITIES ==========")

        for class_name, probability in zip(selected_classes, predictions):
            print(f"{class_name}: {float(probability) * 100:.2f}%")

        print("=============================================")

        

        print("Plant:", plant_name)
        print("Prediction:", predicted_class)
        print("Confidence:", round(confidence, 2), "%")

        image_file.seek(0)

        PredictionHistory.objects.create(
            user=request.user if request.user.is_authenticated else None,
            image=image_file,
            plant=plant,
            predicted_disease=predicted_class,
            confidence=round(confidence, 2),
        )

        return Response({
            "message": "Image analyzed successfully.",
            "filename": image_file.name,
            "prediction_status": "success",
            "plant": plant_name,
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

    serializer_class = PredictionHistorySerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return PredictionHistory.objects.filter(
            user=self.request.user
        ).order_by("-prediction_date")


# =====================================

# =====================================
# PLANT WEATHER RISK CALCULATION
# =====================================

def calculate_plant_weather_risk(
    plant,
    disease,
    temperature,
    humidity,
    rain_probability,
    uv_index
):
    score = 0
    reasons = []
    recommendations = []

    aloe_fungal = [
        "Aloe_Anthracnose",
        "Aloe_Leafspot",
        "Aloe_Rust"
    ]

    tomato_moisture_diseases = [
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato__Target_Spot"
    ]

    tomato_viral_diseases = [
        "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Tomato__Tomato_mosaic_virus"
    ]

    # =====================================
    # ALOE VERA
    # =====================================

    if plant == "aloe":

        if disease in aloe_fungal:
            if humidity is not None and humidity >= 80:
                score += 2
                reasons.append("Very high humidity may favor fungal problems.")
            elif humidity is not None and humidity >= 70:
                score += 1
                reasons.append("High humidity requires leaf monitoring.")

            if rain_probability is not None and rain_probability >= 60:
                score += 2
                reasons.append("Rain may increase leaf moisture.")
            elif rain_probability is not None and rain_probability >= 40:
                score += 1
                reasons.append("Rain is possible. Monitor leaf moisture.")

            if temperature is not None and 18 <= temperature <= 32:
                score += 1
                reasons.append("Temperature may favor some fungal diseases.")

            recommendations.extend([
                "Keep Aloe Vera leaves dry.",
                "Avoid overhead watering.",
                "Provide good airflow.",
                "Inspect leaves for new spots."
            ])

        elif disease == "Aloe_Sunburn":
            if temperature is not None and temperature >= 35:
                score += 2
                reasons.append("High temperature may increase heat stress.")

            if uv_index is not None and uv_index >= 8:
                score += 2
                reasons.append("Very high UV exposure may increase sunburn risk.")
            elif uv_index is not None and uv_index >= 6:
                score += 1
                reasons.append("High UV exposure may stress leaves.")

            recommendations.extend([
                "Protect the plant from intense afternoon sunlight.",
                "Avoid sudden sunlight exposure changes.",
                "Monitor leaves for scorched patches."
            ])

        else:
            if temperature is not None and (
                temperature >= 35 or temperature <= 10
            ):
                score += 1
                reasons.append("Temperature may cause environmental stress.")

            if humidity is not None and humidity >= 80:
                score += 1
                reasons.append("High humidity requires careful watering.")

            recommendations.extend([
                "Maintain suitable watering.",
                "Ensure proper pot drainage.",
                "Monitor leaves regularly."
            ])

    # =====================================
    # TOMATO
    # =====================================

    elif plant == "tomato":

        if disease in tomato_moisture_diseases:

            if humidity is not None and humidity >= 85:
                score += 2
                reasons.append(
                    "Very high humidity may favor leaf diseases."
                )
            elif humidity is not None and humidity >= 70:
                score += 1
                reasons.append(
                    "High humidity may increase favorable conditions."
                )

            if rain_probability is not None and rain_probability >= 60:
                score += 2
                reasons.append(
                    "High rainfall probability may increase leaf wetness."
                )
            elif rain_probability is not None and rain_probability >= 40:
                score += 1
                reasons.append(
                    "Rain is possible. Monitor leaves for moisture."
                )

            if temperature is not None and 15 <= temperature <= 30:
                score += 1
                reasons.append(
                    "Temperature may be favorable for some tomato diseases."
                )

            recommendations.extend([
                "Avoid overhead watering.",
                "Water near the soil instead of leaves.",
                "Provide sufficient spacing and airflow.",
                "Remove infected leaves carefully.",
                "Inspect leaves regularly for new spots."
            ])

        elif disease == "Tomato_Spider_mites_Two_spotted_spider_mite":

            if temperature is not None and temperature >= 30:
                score += 2
                reasons.append(
                    "Hot weather may favor spider mite activity."
                )
            elif temperature is not None and temperature >= 27:
                score += 1
                reasons.append(
                    "Warm weather requires regular pest monitoring."
                )

            if humidity is not None and humidity <= 40:
                score += 2
                reasons.append(
                    "Dry air may favor spider mite activity."
                )
            elif humidity is not None and humidity <= 50:
                score += 1
                reasons.append(
                    "Relatively dry conditions require leaf inspection."
                )

            recommendations.extend([
                "Inspect the undersides of leaves for spider mites.",
                "Look for fine webbing and tiny feeding spots.",
                "Maintain consistent plant watering.",
                "Monitor plants during hot and dry weather."
            ])

        elif disease in tomato_viral_diseases:

            recommendations.extend([
                "Inspect leaves for curling, mosaic patterns and yellowing.",
                "Monitor plants for whiteflies and other insect vectors.",
                "Remove and isolate suspected infected plants appropriately.",
                "Use clean gardening tools."
            ])

            reasons.append(
                "Weather alone cannot determine viral disease risk."
            )

        else:
            if temperature is not None and (
                temperature >= 35 or temperature <= 10
            ):
                score += 1
                reasons.append(
                    "Temperature may cause stress to tomato plants."
                )

            if humidity is not None and humidity >= 85:
                score += 1
                reasons.append(
                    "Very high humidity requires good airflow."
                )

            recommendations.extend([
                "Maintain consistent soil moisture.",
                "Avoid waterlogging.",
                "Provide suitable sunlight.",
                "Monitor leaves and stems regularly."
            ])

    # =====================================
    # RISK LEVEL
    # =====================================

    if score >= 4:
        risk_level = "High"
    elif score >= 2:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    if not reasons:
        reasons.append(
            "Current weather indicators do not show strong risk conditions."
        )

    return {
        "risk_level": risk_level,
        "risk_score": score,
        "reasons": reasons,
        "recommendations": recommendations
    }


# =====================================
# SEASON DETERMINATION
# =====================================

def determine_season(latitude, country="", month=None):
    """
    Determines the current agricultural & meteorological season
    taking into account country/region and hemisphere.
    """
    if month is None:
        month = datetime.now().month

    country_lower = (country or "").lower()
    is_south_asia = any(c in country_lower for c in ["india", "pakistan", "bangladesh", "sri lanka", "nepal"])

    if is_south_asia:
        if month in [12, 1, 2]:
            return {
                "name": "Winter",
                "local_name": "Shishir / Hemant",
                "icon": "❄️",
                "tag": "Cool Season",
                "description": "Cool dry days with cold nights. Plant growth naturally slows down.",
                "care_tip": "Reduce watering frequency, safeguard plants from night chills, and maximize bright morning sun exposure."
            }
        elif month in [3, 4, 5]:
            return {
                "name": "Summer",
                "local_name": "Grishma / Vasant",
                "icon": "☀️",
                "tag": "Warm Season",
                "description": "Rising temperatures and strong midday solar radiation.",
                "care_tip": "Provide afternoon shade to prevent sunburn and hydrate pots early in the morning."
            }
        elif month in [6, 7, 8, 9]:
            return {
                "name": "Monsoon / Rainy",
                "local_name": "Varsha",
                "icon": "🌧️",
                "tag": "Wet Season",
                "description": "Elevated air humidity, frequent rainfall, and cloudy spells.",
                "care_tip": "Ensure pots have free-draining soil; avoid standing water on leaves to suppress fungal spore germination."
            }
        else: # 10, 11
            return {
                "name": "Autumn / Post-Monsoon",
                "local_name": "Sharad",
                "icon": "🍂",
                "tag": "Transition Season",
                "description": "Pleasant transition period with moderate daytime heat and cooling evenings.",
                "care_tip": "Inspect leaves regularly for seasonal rust or spots and regulate watering as ambient moisture drops."
            }

    # Global Northern vs Southern hemisphere
    is_southern = (latitude is not None and latitude < 0)

    if not is_southern:
        # Northern Hemisphere
        if month in [12, 1, 2]:
            return {
                "name": "Winter",
                "local_name": "Dormancy Period",
                "icon": "❄️",
                "tag": "Cold Season",
                "description": "Cold weather with shorter daylight hours and slower plant metabolism.",
                "care_tip": "Shield plants from freezing temperatures, reduce watering, and keep in bright spots."
            }
        elif month in [3, 4, 5]:
            return {
                "name": "Spring",
                "local_name": "Active Growth Season",
                "icon": "🌱",
                "tag": "Growth Season",
                "description": "Warming weather, lengthening days, and vigorous vegetative development.",
                "care_tip": "Ideal period for potting, mild nutrient feeding, and consistent moisture monitoring."
            }
        elif month in [6, 7, 8]:
            return {
                "name": "Summer",
                "local_name": "Peak Sun Season",
                "icon": "☀️",
                "tag": "Warm Season",
                "description": "High temperatures and strong direct UV radiation.",
                "care_tip": "Shield sensitive leaves from scorched spots, deep water early in the day, and maintain soil aeration."
            }
        else: # 9, 10, 11
            return {
                "name": "Autumn / Fall",
                "local_name": "Cooling Transition",
                "icon": "🍂",
                "tag": "Transition Season",
                "description": "Mild temperatures gradually cooling down as sunlight duration shortens.",
                "care_tip": "Prepare plants for cooler days, clear old or yellowed foliage, and monitor humidity."
            }
    else:
        # Southern Hemisphere
        if month in [6, 7, 8]:
            return {
                "name": "Winter",
                "local_name": "Dormancy Period",
                "icon": "❄️",
                "tag": "Cold Season",
                "description": "Cool conditions and shorter days in the southern hemisphere.",
                "care_tip": "Protect leaves from cold drafts and water conservatively."
            }
        elif month in [9, 10, 11]:
            return {
                "name": "Spring",
                "local_name": "Active Growth Season",
                "icon": "🌱",
                "tag": "Growth Season",
                "description": "Spring warming triggering fresh leaf growth and root activity.",
                "care_tip": "Resume regular watering routines and ensure bright indirect light."
            }
        elif month in [12, 1, 2]:
            return {
                "name": "Summer",
                "local_name": "Peak Sun Season",
                "icon": "☀️",
                "tag": "Warm Season",
                "description": "Southern summer heat and high evaporation rates.",
                "care_tip": "Prevent soil from parching and provide shade during peak midday hours."
            }
        else: # 3, 4, 5
            return {
                "name": "Autumn / Fall",
                "local_name": "Cooling Transition",
                "icon": "🍂",
                "tag": "Transition Season",
                "description": "Cooling temperatures as summer recedes.",
                "care_tip": "Moderate watering to avoid waterlogging and watch for damp-soil fungal issues."
            }


# =====================================
# WEATHER BY CITY - ALOE + TOMATO
# =====================================

@api_view(["GET"])
def weather_by_city(request):

    city = request.GET.get("city", "").strip()
    plant = request.GET.get("plant", "aloe").strip().lower()

    if not city:
        return Response(
            {"error": "Please enter a city name."},
            status=status.HTTP_400_BAD_REQUEST
        )

    if plant not in ["aloe", "tomato"]:
        return Response(
            {"error": "Select Aloe Vera or Tomato."},
            status=status.HTTP_400_BAD_REQUEST
        )

    if plant == "aloe":
        allowed_diseases = CLASS_NAMES
        default_disease = "Aloe_Healthy"
        plant_name = "Aloe Vera"
    else:
        allowed_diseases = TOMATO_CLASS_NAMES
        default_disease = "Tomato_healthy"
        plant_name = "Tomato"

    disease = request.GET.get("disease", default_disease).strip()

    if disease not in allowed_diseases:
        return Response(
            {
                "error": f"Unsupported {plant_name} disease class.",
                "supported_classes": allowed_diseases
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    try:

        # 1. FIND CITY COORDINATES

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
                {"error": "City not found. Please enter a valid city name."},
                status=status.HTTP_404_NOT_FOUND
            )

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # 2. GET WEATHER DATA

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

        # 3. CURRENT PLANT CARE

        recommendations = []

        if plant == "aloe":

            if temperature >= 35:
                recommendations.append(
                    "Protect Aloe Vera from extreme afternoon heat."
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
                    "Humidity is high. Avoid overwatering and maintain airflow."
                )

            if rain_probability is not None and rain_probability >= 60:
                recommendations.append(
                    "Rain is likely. Protect Aloe Vera from excess rainfall."
                )

            if uv_index is not None and uv_index >= 6:
                recommendations.append(
                    "UV levels are high. Avoid sudden intense sunlight exposure."
                )

        else:

            if temperature >= 35:
                recommendations.append(
                    "High temperature detected. Provide suitable shade during extreme afternoon heat."
                )
            elif temperature <= 10:
                recommendations.append(
                    "Low temperature detected. Protect tomato plants from cold stress."
                )
            else:
                recommendations.append(
                    "Temperature is suitable for regular tomato plant monitoring."
                )

            if humidity >= 80:
                recommendations.append(
                    "High humidity may favor leaf diseases. Maintain airflow and avoid wet foliage."
                )

            if rain_probability is not None and rain_probability >= 60:
                recommendations.append(
                    "Rain is likely. Check soil drainage and avoid unnecessary watering."
                )

            if uv_index is not None and uv_index >= 8:
                recommendations.append(
                    "Very high UV exposure. Monitor plants for heat and sun stress."
                )

        # 4. THREE-DAY PLANT WEATHER RISK

        daily_risk = []

        for index in range(len(daily["time"])):

            risk = calculate_plant_weather_risk(
                plant=plant,
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

        # 5. SEASON CALCULATION
        season_info = determine_season(
            latitude=latitude,
            country=location.get("country", "")
        )

        # 6. RESPONSE

        response_data = {
            "status": "success",
            "city": location["name"],
            "country": location.get("country", ""),
            "plant": plant_name,
            "disease": disease,
            "season": season_info,

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

            "care_recommendations": recommendations,

            "weather_disease_risk": {
                "plant": plant_name,
                "disease": disease,
                "method": "Rule-based weather risk estimation",
                "daily_risk": daily_risk
            }
        }

        # Keep old Aloe response key for frontend compatibility.
        if plant == "aloe":
            response_data["aloe_care_recommendations"] = recommendations
        else:
            response_data["tomato_care_recommendations"] = recommendations

        return Response(response_data, status=status.HTTP_200_OK)

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
@ensure_csrf_cookie
@api_view(["GET"])
def csrf_token(request):
    return Response({"message": "CSRF cookie set."})