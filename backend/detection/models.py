from django.db import models


class PredictionHistory(models.Model):
    image = models.ImageField(upload_to='prediction_images/')
    predicted_disease = models.CharField(max_length=100)
    confidence = models.FloatField()
    prediction_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.predicted_disease} ({self.confidence:.2f}) - {self.prediction_date.strftime('%Y-%m-%d %H:%M:%S')}"