import cv2
import numpy as np
import faiss
import pickle
import time
import collections
import torch
import os
from utils.logger import setup_logger
# from face_detectors import detect_mtcnn as detector
from face_detectors import detect_mediapipe as detector

from feature_extractors.facenet_extractor import FaceNetEmbedder
from anti_spoofing.scr.deepPixBiS_model import DeepPiXBiS
from preprocess import preprocess_image  # your custom preprocessing

logger = setup_logger()

# Load FAISS index and label map
faiss_index = faiss.read_index("faiss_index/index.bin")
with open("faiss_index/labels.pkl", "rb") as f:
    label_map = pickle.load(f)

embedder = FaceNetEmbedder()
THRESHOLD = 1.0  # For face recognition

# Anti-spoofing model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
anti_spoof_model = DeepPiXBiS().to(device)
anti_spoof_model.load_state_dict(torch.load(r"C:\Users\soura\OneDrive\Desktop\Projects\Face-Detection-and-Recognition\anti_spoofing\model\deepPixBiS_v3.pth", map_location=device))
anti_spoof_model.eval()

def is_real_face(face_crop):
    tensor = preprocess_image(face_crop).to(device)

    with torch.no_grad():
        _, global_output = anti_spoof_model(tensor)
        prediction = torch.sigmoid(global_output).item()
    
    return prediction > 0.5  # True if real, False if spoof

def recognize_face(face_crop):
    emb = embedder.get_embedding(face_crop).astype("float32")
    emb = np.expand_dims(emb, axis=0)  # [1, 512]

    D, I = faiss_index.search(emb, k=1)
    dist, idx = D[0][0], I[0][0]

    if dist < THRESHOLD:
        return label_map[idx], dist
    else:
        return "Unknown", dist

def run_inference(video_source=0):
    fps_history = collections.deque(maxlen=10)
    cap = cv2.VideoCapture(video_source)
    ## seting video writer
    output_path = "output_video/inference_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not cap.isOpened():
        logger.error("Could not open video source.")
        return

    logger.info("Running real-time inference... Press 'q' to quit.")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detect_start = time.time()
        boxes = detector.detect_faces(frame)
        detect_time = (time.time() - detect_start) * 1000  # in milliseconds

        match_time = 0  # Initialize match time

        for (x1, y1, x2, y2) in boxes:
            margin = 0.2
            width, height = x2 - x1, y2 - y1
            x1_e = max(int(x1 - margin * width), 0)
            y1_e = max(int(y1 - margin * height), 0)
            x2_e = min(int(x2 + margin * width), frame.shape[1])
            y2_e = min(int(y2 + margin * height), frame.shape[0])

            face_crop = frame[y1_e:y2_e, x1_e:x2_e]
            if face_crop.size == 0:
                continue
            
            match_start = time.time()

            if is_real_face(face_crop):
                name, dist = recognize_face(face_crop)
                label = f"{name} ({dist:.2f})" if name != "Unknown" else "Unknown"
                color = (0, 255, 0)
            else:
                label = "Spoof Detected"
                color = (0, 0, 255)

            match_time = (time.time()  - match_start) * 1000 # in milliseconds

            cv2.rectangle(frame, (x1_e, y1_e), (x2_e, y2_e), color, 2)
            cv2.putText(frame, label, (x1_e, y1_e - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ## FPS  calculation
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
        else:
            fps = 0.0

        logger.info(f"Frame {frame_count}: Detected {len(boxes)} faces, "
                    f"Detection Time: {detect_time:.2f} ms, "
                    f"Match Time: {match_time:.2f} ms, "
                    f"FPS: {fps:.2f}")  

        # === Display Stats ===
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        cv2.putText(frame, f"Detection Time: {detect_time:.1f} ms", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)
        cv2.putText(frame, f"Matching Time: {match_time:.1f} ms", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)

        cv2.imshow("Face Recognition + Anti-Spoofing", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info(f"Output video saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Video path or 0 for webcam", default=0)
    args = parser.parse_args()

    video_source = args.video if args.video != "0" else 0
    run_inference(video_source)
