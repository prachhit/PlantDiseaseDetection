from django.db import models
from django.contrib.auth.models import User

class DiseaseInfo(models.Model):
    """Detailed info for the Results page, managed via Admin."""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField()
    symptoms = models.TextField()
    treatment = models.TextField()
    prevention = models.TextField()

    def __str__(self):
        return self.name

class DiseasePrediction(models.Model):
    """Logs every scan made by a user."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to='prediction_logs/')
    predicted_class = models.CharField(max_length=100)
    confidence = models.FloatField()
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.predicted_class} - {self.date.strftime('%Y-%m-%d')}"