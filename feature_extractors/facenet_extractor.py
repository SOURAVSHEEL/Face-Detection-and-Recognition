import torch
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image

class FaceNetEmbedder:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    def preprocess(self, face):
        """
        Args:
            face (np.ndarray): BGR image from OpenCV
        Returns:
            torch.Tensor: [1, 3, 224, 224]
        """
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        return tensor

    def get_embedding(self, face):
        """
        Args:
            face (np.ndarray): cropped face (BGR)
        Returns:
            np.ndarray: 512-d face embedding
        """
        with torch.no_grad():
            input_tensor = self.preprocess(face)
            embedding = self.model(input_tensor)
        return embedding.squeeze().cpu().numpy()
