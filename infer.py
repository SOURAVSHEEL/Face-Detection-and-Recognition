import cv2
import numpy as np
import faiss
import pickle
import time
import collections

from utils.logger import setup_logger
# from face_detectors import  detect_mtcnn as detector
from face_detectors import detect_mediapipe as detector

from feature_extractors.facenet_extractor import FaceNetEmbedder

logger = setup_logger()

# Load FAISS index and label map
faiss_index = faiss.read_index("faiss_index/index.bin")
with open("faiss_index/labels.pkl", "rb") as f:
    label_map = pickle.load(f)

embedder = FaceNetEmbedder()
THRESHOLD = 1.0  # Adjust based on testing

def recognize_face(face_crop):
    emb = embedder.get_embedding(face_crop).astype("float32")
    emb = np.expand_dims(emb, axis=0)  # [1, 512]

    D, I = faiss_index.search(emb, k=1)  # nearest neighbor
    dist, idx = D[0][0], I[0][0]

    if dist < THRESHOLD:
        return label_map[idx], dist
    else:
        return "Unknown", dist

def run_inference(video_source=0):
    fps_history = collections.deque(maxlen=10)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error("Could not open video source.")
        return

    logger.info("Running inference... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        boxes = detector.detect_faces(frame)
        detect_time = time.time() - start

        for (x1, y1, x2, y2) in boxes:
            # Expand slightly
            margin = 0.2
            width, height = x2 - x1, y2 - y1
            x1_e = max(int(x1 - margin * width), 0)
            y1_e = max(int(y1 - margin * height), 0)
            x2_e = min(int(x2 + margin * width), frame.shape[1])
            y2_e = min(int(y2 + margin * height), frame.shape[0])

            face_crop = frame[y1_e:y2_e, x1_e:x2_e]
            if face_crop.size == 0:
                continue

            name, dist = recognize_face(face_crop)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({dist:.2f})" if name != "Unknown" else "Unknown"

            cv2.rectangle(frame, (x1_e, y1_e), (x2_e, y2_e), color, 2)
            cv2.putText(frame, label, (x1_e, y1_e - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        fps = 1.0 / detect_time if detect_time > 0 else 0.0
        # Add to FPS history
        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition - Press 'q' to quit", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Path to video file (or leave empty for webcam)", default=0)
    args = parser.parse_args()

    video_source = args.video if args.video != "0" else 0
    run_inference(video_source)
