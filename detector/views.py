# detector/views.py (Focus on the logging part)

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .utils import predict_image
from .models import DiseasePrediction # Make sure this is imported

def detection_view(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('image_file'):
        uploaded_file = request.FILES['image_file']
        
        # Save the file (critical for prediction AND logging)
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename) # Full OS path for ML model

        # 1. Get prediction
        predicted_class, confidence = predict_image(file_path)

        # 2. Log Prediction to Database (Admin Tracking)
        DiseasePrediction.objects.create(
            # The .name attribute gives the path relative to MEDIA_ROOT
            image=filename, 
            predicted_class=predicted_class,
            confidence=confidence
        )

        context = {
            'predicted_class': predicted_class,
            'confidence': f"{confidence * 100:.2f}%",
            'image_url': fs.url(filename) # URL for the template
        }
        return render(request, 'detector/result.html', context)
    
    return render(request, 'detector/upload.html')
    # In detector/views.py (inside detection_view POST logic)
    MAX_UPLOAD_SIZE = 5242880 # 5MB in bytes

    if request.FILES['image_file'].size > MAX_UPLOAD_SIZE:
        # Handle the error gracefully
        return render(request, 'detector/upload.html', {'error_message': 'File size exceeds 5MB limit.'})
    # ... continue with fs.save and prediction