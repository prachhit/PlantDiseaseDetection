# detector/utils.py

import os
import numpy as np
from PIL import Image
from django.conf import settings
import tensorflow as tf

# Load the model globally when Django starts, not on every request
MODEL_PATH = os.path.join(settings.BASE_DIR, 'detector', 'ml_models', 'plant_disease_model.h5')
try:
    PLANT_MODEL = tf.keras.models.load_model(MODEL_PATH)
    # Define class names (make sure this list matches your model's output order)
    CLASS_NAMES = ['Tomato_Healthy', 'Tomato__Early_Blight', 'Pepper__Bacterial_Spot', 'Corn__Rust', 'Potato__Late_Blight'] 
except Exception as e:
    print(f"Error loading ML model: {e}")
    PLANT_MODEL = None
    CLASS_NAMES = []


def predict_image(image_path):
    """
    Loads an image, preprocesses it, and makes a prediction using the model.
    """
    if PLANT_MODEL is None:
        return "Model Not Loaded", 0.0

    # 1. Load and Preprocess Image
    img = Image.open(image_path).convert('RGB')
    # Resize to the model's expected input size (e.g., 224x224)
    img = img.resize((224, 224)) 
    img_array = np.array(img)
    # Normalize to [0, 1]
    img_array = img_array / 255.0 
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0) 

    # 2. Predict
    predictions = PLANT_MODEL.predict(img_batch)[0]
    
    # 3. Get Result
    confidence = np.max(predictions)
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence