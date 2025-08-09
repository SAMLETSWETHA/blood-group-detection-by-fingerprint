import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("blood_group_model.h5")

# Define blood group labels (Must match your training labels)
blood_groups = ["A-", "A+", "AB-", "AB+", "B-", "B+", "O-", "O+"]

# Image path for testing
test_image_path = r"C:\\Users\\Admin\\Downloads\\archive (2)\\processed\\cluster_0_16.BMP.jpg"

# Function to preprocess the image
def prepare_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    if img is None:
        raise FileNotFoundError(f"Error: Could not load image at {image_path}. Check the file path.")

    img = cv2.resize(img, (128, 128))  # Resize to match model input shape
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (128, 128, 1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 128, 128, 1)
    
    return img

# Prepare the image
image = prepare_image(test_image_path)

# Make prediction
prediction = model.predict(image)

# Decode the prediction
predicted_index = np.argmax(prediction)
predicted_label = blood_groups[predicted_index]

print(f" 🩸 Predicted Blood Group: {predicted_label}")