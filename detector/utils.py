import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Ensure these match your DISEASE_INFO keys
DISEASE_INFO = {
    "Bacterialblight": {"description": "Bacterial infection causing wilting.", "symptoms": "Water-soaked streaks.", "treatment": "Copper-based bactericides."},
    "Blast": {"description": "Fungal disease (rice).", "symptoms": "Diamond-shaped spots.", "treatment": "Apply Tricyclazole."},
    "Brownspot": {"description": "Fungal disease from nutrient stress.", "symptoms": "Oval brown spots.", "treatment": "Improve Potassium."},
    "Tungro": {"description": "Viral disease spread by leafhoppers.", "symptoms": "Stunting/orange leaves.", "treatment": "Control leafhoppers."},
    "Bacterial spot": {"description": "Common in moist environments.", "symptoms": "Small necrotic spots.", "treatment": "Copper fungicides."},
    "Black mold": {"description": "Sooty fungal growth.", "symptoms": "Black powdery coating.", "treatment": "Control insects/fungicide."},
    "Gray spot": {"description": "Fungal leaf pathogen.", "symptoms": "Rectangular gray lesions.", "treatment": "Apply Mancozeb."},
    "health": {"description": "Plant is healthy.", "symptoms": "None.", "treatment": "Normal care."},
    "Late blight": {"description": "Phytophthora infestans.", "symptoms": "Dark water-soaked patches.", "treatment": "Immediate fungicide."},
    "powdery mildew": {"description": "White flour-like patches.", "symptoms": "White powdery coating.", "treatment": "Sulfur/Neem oil."},
    "Bacterial Leaf Streak": {"description": "Bacterial infection.", "symptoms": "Linear lesions.", "treatment": "Copper spray."},
    "Common_rust": {"description": "Fungal rust spores.", "symptoms": "Orange-brown pustules.", "treatment": "Fungicide application."},
    "Gray_leaf_spot": {"description": "Rectangular necrotic spots.", "symptoms": "Gray lesions on leaves.", "treatment": "Fungicide."},
    "Healthy": {"description": "Plant is in good health.", "symptoms": "Vibrant foliage.", "treatment": "No medicine needed."},
    "Maize Chlorotic Mottle Virus": {"description": "Severe viral disease.", "symptoms": "Yellowing and mottling.", "treatment": "Remove infected plants."}
}

def is_mostly_green(image):
    """Checks if the image has enough green pixels to likely be a leaf."""
    # Convert to HSV (Hue, Saturation, Value) for better color detection
    hsv_img = image.convert('HSV')
    np_img = np.array(hsv_img)
    
    h = np_img[:, :, 0] # Hue
    s = np_img[:, :, 1] # Saturation
    v = np_img[:, :, 2] # Value
    
    # Define green range (Hue roughly 35-95, with minimum saturation/brightness)
    green_mask = (h > 30) & (h < 100) & (s > 30) & (v > 30)
    green_count = np.sum(green_mask)
    total_pixels = green_mask.size
    green_ratio = green_count / total_pixels
    
    return green_ratio > 0.15  # Returns True if at least 15% of the image is green

def predict_image(image_path):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'ml_models', 'saved_model_folder')
    label_path = os.path.join(BASE_DIR, 'ml_models', 'labels.txt')

    # 1. Open and check color FIRST
    image = Image.open(image_path).convert("RGB")
    
    # --- ADDED COLOR FILTER ---
    if not is_mostly_green(image):
        return {
            "label": "NON_PLANT", 
            "confidence": "0.00",
            "description": "The image does not contain enough green to be identified as a leaf.",
            "symptoms": "N/A",
            "treatment": "N/A"
        }

    # 2. Load Model & Labels
    model = tf.saved_model.load(model_path)
    infer = model.signatures["serving_default"]
    
    with open(label_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # 3. Image Processing
    processed_image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = (np.asarray(processed_image).astype(np.float32) / 127.5) - 1
    input_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0))

    # 4. Prediction
    output = infer(input_tensor)
    output_key = list(output.keys())[0]
    prediction = output[output_key].numpy()[0]
    
    index = np.argmax(prediction)
    raw_label = class_names[index]
    clean_label = raw_label.split(' ', 1)[-1].strip()
    confidence = float(prediction[index])

    # 5. STRICT AI FILTER
    sorted_scores = np.sort(prediction)[::-1]
    margin = sorted_scores[0] - sorted_scores[1]

    # If the AI is not confident OR the top two guesses are too close
    if confidence < 0.85 or margin < 0.20:
        return {
            "label": "NON_PLANT", 
            "confidence": "0.00",
            "description": "AI is unsure. This might not be a plant.",
            "symptoms": "N/A",
            "treatment": "N/A"
        }

    info = DISEASE_INFO.get(clean_label, {
        "description": "Detected plant.", "symptoms": "N/A", "treatment": "N/A"
    })

    return {
        "label": clean_label,
        "confidence": f"{confidence * 100:.2f}%",
        "description": info["description"],
        "symptoms": info["symptoms"],
        "treatment": info["treatment"]
    }