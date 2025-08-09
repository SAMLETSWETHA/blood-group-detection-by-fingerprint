from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pickle

# Load the trained model
model = load_model("blood_group_model.h5")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def predict_blood_group(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((128, 128))  # Resize as per your model's input shape
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 128, 128, 1)  # Adjust dimensions

    prediction = model.predict(img_array)
    labels = ["A+","A-","B+", "B-","AB+","AB-","O+","O-"]  # Adjust labels based on training
    return labels[np.argmax(prediction)]