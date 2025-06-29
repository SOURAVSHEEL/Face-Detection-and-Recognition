from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import cv2


# Preprocess image to match the input requirements of the model
# Resize to 224x224, convert to RGB, normalize, and convert to tensor   

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # ensure same size as training
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    transform = transforms.Compose([
        transforms.ToTensor(),                    # HWC to CHW and [0,255] -> [0.0, 1.0]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])     # Match training normalization
    ])
    tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]
    return tensor
