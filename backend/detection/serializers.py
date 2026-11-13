from rest_framework import serializers
from .models import PredictionHistory


class PredictionHistorySerializer(serializers.ModelSerializer):

    plant_display = serializers.CharField(
        source="get_plant_display",
        read_only=True
    )

    class Meta:
        model = PredictionHistory
        fields = [
            "id",
            "image",
            "plant",
            "plant_display",
            "predicted_disease",
            "confidence",
            "prediction_date"
        ]