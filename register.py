import cv2
import os
import argparse
from datetime import datetime
from utils.logger import setup_logger

# Face detection using MediaPipe
from face_detectors import detect_mediapipe as detector

# Anti-spoofing model
import torch
from anti_spoofing.scr.deepPixBiS_model import DeepPiXBiS
from preprocess import preprocess_image

logger = setup_logger()

# Load anti-spoofing model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spoof_model = DeepPiXBiS().to(device)
spoof_model.load_state_dict(torch.load(r"C:\Users\soura\OneDrive\Desktop\Projects\Face-Detection-and-Recognition\anti_spoofing\model\deepPixBiS_v2.pth", map_location=device))
spoof_model.eval()

def is_real_face(face_crop, threshold=0.4):
    face_tensor = preprocess_image(face_crop).to(device)
    with torch.no_grad():
        _, global_pred = spoof_model(face_tensor)
        raw = torch.sigmoid(global_pred).item()
        return raw > threshold, raw

def register(name):
    save_dir = os.path.join("data", name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    logger.info(f"Registering new identity: {name}")
    logger.info("Press 's' to save real face | 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read from camera.")
            break

        boxes = detector.detect_faces(frame)

        for (x1, y1, x2, y2) in boxes:
            margin = 0.1
            width, height = x2 - x1, y2 - y1
            x1_exp = max(int(x1 - margin * width), 0)
            y1_exp = max(int(y1 - margin * height), 0)
            x2_exp = min(int(x2 + margin * width), frame.shape[1])
            y2_exp = min(int(y2 + margin * height), frame.shape[0])
            face_crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]

            if face_crop.size == 0:
                continue

            real, score = is_real_face(face_crop)

            color = (0, 255, 0) if real else (0, 0, 255)
            label = f"Real ({score:.2f})" if real else f"Fake ({score:.2f})"

            cv2.rectangle(frame, (x1_exp, y1_exp), (x2_exp, y2_exp), color, 2)
            cv2.putText(frame, label, (x1_exp, y1_exp - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Registration - Press 's' to save | 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and boxes:
            saved_any = False
            for (x1, y1, x2, y2) in boxes:
                width, height = x2 - x1, y2 - y1
                x1 = max(int(x1 - 0.1 * width), 0)
                y1 = max(int(y1 - 0.1 * height), 0)
                x2 = min(int(x2 + 0.1 * width), frame.shape[1])
                y2 = min(int(y2 + 0.1 * height), frame.shape[0])
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                real, score = is_real_face(face_crop)
                if real:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{name}_{count}_{timestamp}.jpg"
                    save_path = os.path.join(save_dir, filename)
                    cv2.imwrite(save_path, face_crop)
                    logger.info(f"Saved: {save_path} | Score: {score:.2f}")
                    count += 1
                    saved_any = True
                else:
                    logger.warning(f"Spoof face detected (score: {score:.2f}). Not saved.")

            if not saved_any:
                logger.warning("No real face detected to save.")

        elif key == ord('q'):
            logger.info("Exiting registration.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Registration Tool")
    parser.add_argument("--name", required=True, help="Name of the person to register")
    args = parser.parse_args()
    register(args.name)
