from django.shortcuts import render
from .utils import predict_image
import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage

def detection_view(request):
    if request.method == 'POST' and request.FILES.get('image_file'):
        img_file = request.FILES['image_file']
        fs = FileSystemStorage()
        filename = fs.save(img_file.name, img_file)
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        uploaded_file_url = fs.url(filename)
        
        # Get data from your utils.py
        res = predict_image(file_path)
        
        # DEBUG: Check your terminal to see if these print!
        print(f"PREDICTION FOUND: {res['label']} - {res['confidence']}")

        # These keys MUST match your HTML tags exactly
        return render(request, 'detector/result.html', {
            'label': res['label'],
            'confidence': res['confidence'],
            'description': res['description'],
            'symptoms': res['symptoms'],
            'treatment': res['treatment'],
            'image_url': uploaded_file_url,
        })
    
    return render(request, 'detector/upload.html')