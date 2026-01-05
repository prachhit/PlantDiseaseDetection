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

def is_valid_plant_image(image):
    """
    Analyzes the image to see if it contains enough green/organic 
    color to be a plant leaf.
    """
    # Convert to HSV which is better for color isolation
    hsv_img = image.convert('HSV')
    np_img = np.array(hsv_img)
    
    h = np_img[:, :, 0] # Hue
    s = np_img[:, :, 1] # Saturation
    v = np_img[:, :, 2] # Value
    
    # 1. Define the Green Range (Hue: 30-90 is generally green)
    # 2. Check Saturation (Plants are vibrant, charts are dull/grey)
    green_mask = (h > 35) & (h < 95) & (s > 40) & (v > 30)
    
    green_count = np.sum(green_mask)
    total_pixels = green_mask.size
    green_ratio = green_count / total_pixels
    
    # If the image is less than 15% green, it's likely not a leaf
    return green_ratio > 0.15

def predict_image(image_path):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'ml_models', 'saved_model_folder')
    label_path = os.path.join(BASE_DIR, 'ml_models', 'labels.txt')

    # 1. Open image and check color FIRST
    original_image = Image.open(image_path).convert("RGB")
    
    if not is_valid_plant_image(original_image):
        return {
            "label": "NON_PLANT", 
            "confidence": "0.00",
            "description": "Invalid Image",
            "symptoms": "N/A",
            "treatment": "N/A"
        }

    # 2. Load Model & Labels
    model = tf.saved_model.load(model_path)
    infer = model.signatures["serving_default"]
    
    with open(label_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # 3. Process for AI
    image = ImageOps.fit(original_image, (224, 224), Image.Resampling.LANCZOS)
    img_array = (np.asarray(image).astype(np.float32) / 127.5) - 1
    input_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0))

    # 4. Prediction
    output = infer(input_tensor)
    output_key = list(output.keys())[0]
    prediction = output[output_key].numpy()[0]
    
    index = np.argmax(prediction)
    raw_label = class_names[index]
    clean_label = raw_label.split(' ', 1)[-1].strip()
    confidence = float(prediction[index])

    # 5. Final AI Guard - Even if it's green, the AI must be sure
    if confidence < 0.85:
        return {"label": "NON_PLANT", "confidence": "0.00"}

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