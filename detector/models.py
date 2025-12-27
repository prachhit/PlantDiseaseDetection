# detector/models.py
# ... (Keep DiseasePrediction class)
from django.db import models

class DiseaseInfo(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField()
    symptoms = models.TextField()
    treatment = models.TextField()
    prevention = models.TextField()

    def __str__(self):
        return self.name

    class Meta:
        verbose_name_plural = "Disease Information"


class DiseasePrediction(models.Model):
    image = models.ImageField(upload_to='uploads/')
    predicted_class = models.CharField(max_length=100)
    confidence = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.predicted_class} ({self.confidence:.2f}) at {self.timestamp}"