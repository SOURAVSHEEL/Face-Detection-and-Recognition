import cv2
import torch
import numpy as np
from deepPixBiS_model import DeepPiXBiS
from torchvision import transforms

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DeepPiXBiS().to(device)
model.load_state_dict(torch.load(
    r"C:\Users\soura\OneDrive\Desktop\Projects\Face-Detection-and-Recognition\anti_spoofing\model\deepPixBiS.pth",
    map_location=device
))
model.eval()

# Transform for input image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]

        if face.size == 0:
            continue

        face_input = transform(face).unsqueeze(0).to(device)

        # Model prediction
        with torch.no_grad():
            _, global_output = model(face_input)
            prediction = (global_output.item() > 0.5)

        label = "Real" if prediction else "Spoof"
        color = (0, 255, 0) if label == "Real" else (0, 0, 255)

        # Display result
        cv2.rectangle(orig, (x, y), (x + w, y + h), color, 2)
        cv2.putText(orig, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Anti-Spoofing Detection", orig)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
