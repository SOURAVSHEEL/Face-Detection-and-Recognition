import cv2
import os
import argparse
from datetime import datetime
from utils.logger import setup_logger

# ✅ Manually import your face detector here
# from face_detectors import detect_mtcnn as detector
from face_detectors import detect_mediapipe as detector

logger = setup_logger()

def register(name):
    # Setup folder
    save_dir = os.path.join("data", name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    logger.info(f"Registering new identity: {name}")
    logger.info(f"Using face detector: MTCNN")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read from camera.")
            break

        # Detect faces
        boxes = detector.detect_faces(frame)

        for (x1, y1, x2, y2) in boxes:

            ## expand the bounding box slightly
            margin = 0.1
            width = x2 - x1
            height = y2 - y1

            x1_exp = max(int(x1 - margin *  width), 0)
            y1_exp = max(int(y1 - margin * height), 0)
            x2_exp = min(int(x2 + margin * width), frame.shape[1])
            y2_exp = min(int(y2 + margin * height), frame.shape[0])
            x1, y1, x2, y2 = x1_exp, y1_exp, x2_exp, y2_exp
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Registration - Press 's' to save | 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and boxes:
            x1, y1, x2, y2 = boxes[0]

            margin = 0.1
            width = x2 - x1     
            height = y2 - y1
            x1_exp = max(int(x1 - margin * width), 0)   
            y1_exp = max(int(y1 - margin * height), 0)
            x2_exp = min(int(x2 + margin * width), frame.shape[1])
            y2_exp = min(int(y2 + margin * height), frame.shape[0])
            x1, y1, x2, y2 = x1_exp, y1_exp, x2_exp, y2_exp
            ## crop the face         
            face_crop = frame[y1:y2, x1:x2]

            ## skip if cro is invalid
            if face_crop.size == 0:
                logger.warning("Invaild face crop detected, skipping save.")
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{count}_{timestamp}.jpg"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, face_crop)
            logger.info(f"Saved: {save_path}")
            count += 1

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




