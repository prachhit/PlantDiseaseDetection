import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# IMPORTANT: Ensure these keys match your labels.txt EXACTLY
DISEASE_INFO = {
    "Bacterialblight": {
        "description": "A serious bacterial infection that causes wilting of seedlings and yellowing of leaves.",
        "symptoms": "Water-soaked streaks that turn yellow or white; droplets of bacterial ooze on leaves.",
        "treatment": "Use balanced nitrogen. Apply Copper-based bactericides or Streptomycin sulfate."
    },
    "Blast": {
        "description": "One of the most destructive rice diseases, caused by the fungus Magnaporthe oryzae.",
        "symptoms": "Diamond-shaped (spindle) spots with gray centers and brown margins.",
        "treatment": "Apply Tricyclazole 75 WP or Carbendazim. Avoid excessive nitrogen fertilizer."
    },
    "Brownspot": {
        "description": "A fungal disease often linked to nutrient-deficient soil or water-stressed plants.",
        "symptoms": "Small, oval, dark brown spots spread across the leaf surface.",
        "treatment": "Apply Mancozeb or Iprobenfos. Improve soil quality by adding Potassium."
    },
    "Tungro": {
        "description": "A viral disease spread by green leafhoppers.",
        "symptoms": "Stunting of the plant and leaves turning orange or yellow.",
        "treatment": "No direct medicine for the virus. Use insecticides to control leafhoppers."
    },
    "Bacterial spot": {
        "description": "Bacterial infection common in warm, moist environments.",
        "symptoms": "Small, dark, water-soaked spots that eventually turn necrotic.",
        "treatment": "Apply Copper-based fungicides. Avoid overhead irrigation."
    },
    "Late blight": {
        "description": "A fast-spreading disease caused by Phytophthora infestans.",
        "symptoms": "Large, irregular dark green to black water-soaked patches on leaves.",
        "treatment": "Apply fungicides like Chlorothalonil or Mancozeb immediately."
    },
    "powdery mildew": {
        "description": "Fungal disease that looks like white flour dusted on leaves.",
        "symptoms": "White powdery patches on the upper surface of leaves.",
        "treatment": "Apply Sulfur-based fungicides or Neem oil."
    },
    "health": {
        "description": "The plant shows no signs of disease infection.",
        "symptoms": "Normal green leaves, no spots or discoloration.",
        "treatment": "No medicine needed. Continue standard agricultural practices."
    },
    "Healthy": {
        "description": "The plant is in good health.",
        "symptoms": "Clean, green, and vibrant foliage.",
        "treatment": "No medicine needed."
    }
    # Note: You can add Bacterial Leaf Streak, Common_rust, etc., following this same format.
}

def predict_image(image_path):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'ml_models', 'saved_model_folder')
    label_path = os.path.join(BASE_DIR, 'ml_models', 'labels.txt')

    # 1. Load Model & Labels
    model = tf.saved_model.load(model_path)
    infer = model.signatures["serving_default"]
    
    with open(label_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # 2. Image Processing
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = (np.asarray(image).astype(np.float32) / 127.5) - 1
    input_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0))

    # 3. Prediction
    output = infer(input_tensor)
    output_key = list(output.keys())[0]
    prediction = output[output_key].numpy()
    
    index = np.argmax(prediction)
    raw_label = class_names[index]
    
    # Cleans "0 Tomato_Healthy" to "Tomato_Healthy"
    clean_label = raw_label.split(' ', 1)[-1].strip()
    confidence = float(prediction[0][index]) * 100

    # 4. Get Info or Use Default if not found in dictionary
    info = DISEASE_INFO.get(clean_label, {
        "description": f"Detected {clean_label}. Detailed info not in database.",
        "symptoms": "Consult agricultural guide.",
        "treatment": "General fungicide or expert consultation recommended."
    })

    return {
        "label": clean_label,
        "confidence": f"{confidence:.2f}%",
        "description": info["description"],
        "symptoms": info["symptoms"],
        "treatment": info["treatment"]
    }