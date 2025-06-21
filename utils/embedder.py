import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# Load FaceNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Transform image to 160x160 and normalize
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def get_embedding(face_image_bgr):
    """
    Extract embedding from a face image using FaceNet.
    Args:
        face_image_bgr (np.ndarray): Cropped BGR image of face
    Returns:
        embedding (np.ndarray): 512-dim numpy array
    """
    face_rgb = face_image_bgr[:, :, ::-1]  # BGR to RGB
    face_tensor = transform(face_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(face_tensor)

    return embedding.cpu().numpy().flatten()
