from PIL import Image
import numpy as np
import os

def load_image(image_path):
    """
    Load an image from the specified path and return it as a PIL Image object.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image at {image_path} does not exist.")
    return Image.open(image_path)

def preprocess_image(image, target_size=(224, 224)):
    """
    Resize and normalize the image for analysis.
    """
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return image_array

def save_image(image, save_path):
    """
    Save the processed image to the specified path.
    """
    image.save(save_path)

def get_image_paths(directory):
    """
    Retrieve all image file paths from the specified directory.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]