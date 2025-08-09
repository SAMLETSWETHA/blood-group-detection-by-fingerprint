import cv2
import os

# Folder path containing images
folder_path = r"C:\Users\Admin\Downloads\archive (2)\processed"

# List all files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(".BMP.jpg")]

# Check if images are being read correctly
for img_file in image_files:
    img_path = os.path.join(folder_path, img_file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Error: Cannot read {img_file}")
    else:
        print(f"✅ Successfully loaded {img_file}")

print(f"\nTotal images found: {len(image_files)}")
def preprocess_image(image_path):
    # Your image preprocessing code here
    return preprocess_image
