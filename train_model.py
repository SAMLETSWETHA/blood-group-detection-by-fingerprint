import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle
import re  # Import regex for better label extraction

# Define dataset path
DATASET_PATH = r"C:\Users\Admin\Downloads\archive (2)\processed"

# Define blood group labels in a fixed order
blood_groups = ["A-", "A+", "AB-", "AB+", "B-", "B+", "O-", "O+"]

# Function to load images and labels
def load_images(dataset_path):
    images = []
    labels = []
    
    for filename in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
        img = cv2.resize(img, (128, 128))  # Resize to 128x128
        images.append(img)

        # Extract blood group label from filename using regex
        match = re.search(r"_(A\+|A-|AB\+|AB-|B\+|B-|O\+|O-)", filename)
        if match:
            labels.append(match.group(1))  # Extract the blood group label
        else:
            print(f"⚠ Warning: Could not extract label from {filename}")  # Debugging info
            continue

    # Convert lists to NumPy arrays
    images = np.array(images).reshape(-1, 128, 128, 1) / 255.0  # Normalize
    labels = np.array(labels)

    return images, labels

# Load dataset
images, labels = load_images(DATASET_PATH)

# Convert labels to numeric values using a fixed mapping
label_encoder = LabelEncoder()
label_encoder.fit(blood_groups)  # Ensure fixed order for encoding
labels = label_encoder.transform(labels)
labels = to_categorical(labels, num_classes=8)  # Convert to one-hot encoding

# Split data into training (80%) and validation (20%)
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='softmax')  # 8 classes for blood groups
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_images, train_labels, epochs=20, validation_data=(val_images, val_labels))

# Save the trained label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Save Model
model.save("blood_group_model.h5")
print("✅ Model training complete! Saved as blood_group_model.h5")
