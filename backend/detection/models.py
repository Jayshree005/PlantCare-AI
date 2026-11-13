from django.db import models
from django.conf import settings


class PredictionHistory(models.Model):
    PLANT_CHOICES = [
        ("aloe", "Aloe Vera"),
        ("tomato", "Tomato"),
    ]

    image = models.ImageField(upload_to='prediction_images/')

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="plant_predictions",
        null=True,
        blank=True
    )

    plant = models.CharField(
        max_length=20,
        choices=PLANT_CHOICES,
        default="aloe"
    )
    predicted_disease = models.CharField(max_length=100)
    confidence = models.FloatField()
    prediction_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.plant} - {self.predicted_disease} ({self.confidence:.2f}) - {self.prediction_date.strftime('%Y-%m-%d %H:%M:%S')}"