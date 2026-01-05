from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .utils import predict_image
import os

def detection_view(request):
    if request.method == 'GET':
        return render(request, 'detector/upload.html')

    if request.method == 'POST' and request.FILES.get('image_file'):
        uploaded_file = request.FILES['image_file']
        
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        file_url = fs.url(filename)

        try:
            prediction = predict_image(file_path)
            
            # Use the exact label from utils.py
            raw_label = prediction.get('label', 'Unknown')
            label_lower = raw_label.lower()
            confidence = float(str(prediction.get('confidence', '0')).replace('%', '').strip())

            # 1. SPECIAL CASE: Reject the "NON_PLANT" flag from your new utils.py
            if raw_label == "NON_PLANT":
                if os.path.exists(file_path): os.remove(file_path)
                return render(request, 'detector/upload.html', {
                    'error': 'Object not recognized. Please upload a clear leaf photo.'
                })

            # 2. UPDATED KEYWORDS: Must match your 15 specific disease names
            # If the prediction doesn't contain one of these, it's likely a chart/animal
            valid_markers = [
                'blight', 'blast', 'brownspot', 'tungro', 'spot', 'mold', 
                'streak', 'rust', 'mildew', 'virus', 'health'
            ]
            
            is_valid = any(marker in label_lower for marker in valid_markers)

            # 3. STRICT FILTERING
            # We reject if keywords don't match OR if confidence is too low for a generic "Healthy" result
            # Animals usually trigger 'Healthy' or 'health' with high scores, so we set a high bar.
            if not is_valid or (confidence < 85 and 'health' in label_lower):
                if os.path.exists(file_path):
                    os.remove(file_path)
                return render(request, 'detector/upload.html', {
                    'error': 'This does not look like a diseased or healthy leaf. Please try again.'
                })

            # 4. Success - Send to Result
            context = {
                'image_url': file_url,
                'label': raw_label,
                'confidence': confidence,
                'description': prediction.get('description', ''),
                'symptoms': prediction.get('symptoms', ''),
                'treatment': prediction.get('treatment', ''),
            }
            return render(request, 'detector/result.html', context)

        except Exception as e:
            print(f"Error: {e}")
            if os.path.exists(file_path): os.remove(file_path)
            return render(request, 'detector/upload.html', {'error': 'Analysis failed.'})

    return render(request, 'detector/upload.html', {'error': 'Please upload an image.'})