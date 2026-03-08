import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Input and output directories
input_dir = 'dataset/'  # Folder containing medical images
output_dir = 'processed_dataset/'  # Folder to save processed images
os.makedirs(output_dir, exist_ok=True)

# ✅ Data Augmentation
aug_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Process all images in the dataset recursively
for root, _, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            image_path = os.path.join(root, filename)
            relative_path = os.path.relpath(root, input_dir)  # Preserve subdirectory structure
            save_dir = os.path.join(output_dir, relative_path)
            os.makedirs(save_dir, exist_ok=True)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Resize image to 128x128
            image_resized = cv2.resize(image, (128, 128))
            image_resized = np.expand_dims(image_resized, axis=-1)  # Add channel dimension
            image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension

            # Generate augmented images
            i = 0
            for batch in aug_datagen.flow(image_resized, batch_size=1, save_to_dir=save_dir, save_prefix=f"aug_{filename}", save_format='png'):
                i += 1
                if i >= 5:  # Generate 5 augmented images per input
                    break

            print(f"Augmented {filename} -> Saved 5 augmented versions in {save_dir}")

print("Data augmentation complete for all dataset images.")
