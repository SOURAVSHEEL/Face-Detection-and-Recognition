import torch
import cv2
import numpy as np
from torchvision import transforms
from anti_spoofing.scr.deepPixBiS_model import DeepPiXBiS

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DeepPixBiS model
model = DeepPiXBiS().to(device)
model_path = r"C:\Users\soura\OneDrive\Desktop\Projects\Face-Detection-and-Recognition\anti_spoofing\model\deepPixBiS_v2.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define transform: must match training (no flip, yes resize + normalization)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Match training image size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def check_real_face(face_crop):
    """
    Check if a given face crop is real or spoof.

    Args:
        face_crop (np.ndarray): BGR cropped face image from OpenCV.

    Returns:
        bool: True if real face (live), False if spoof.
    """
    if face_crop is None or face_crop.size == 0:
        print("[AntiSpoofing] Invalid face crop.")
        return False

    try:
        # Convert to tensor
        image = transform(face_crop).unsqueeze(0).to(device)

        # Model prediction
        with torch.no_grad():
            _, global_pred = model(image)
            score = torch.sigmoid(global_pred).item()

        print(f"[Anti-Spoofing Score] {score:.4f}")
        return score > 0.2 # Threshold for real face

    except Exception as e:
        print(f"[AntiSpoofing] Error: {e}")
        return False
