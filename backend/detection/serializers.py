from rest_framework import serializers
from .models import PredictionHistory


class PredictionHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionHistory
        fields = [
            'id',
            'image',
            'predicted_disease',
            'confidence',
            'prediction_date'
        ]