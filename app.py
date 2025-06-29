import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from face_detectors import detect_mediapipe as detector
from utils.embedder import get_embedding
from utils.matcher_faiss import add_new_identity, find_match_faiss
from utils.preprocessing import crop_face

st.set_page_config(page_title="Face Recognition", layout="centered")
st.title("Real-time Face Recognition App")
tab = st.sidebar.radio("Choose Mode", ["Register Face", "Recognize Face"], key="mode_selector")

# Webcam Face Registration

if tab == "Register Face":
    st.subheader("Register New Face via Webcam")
    name = st.text_input("Enter name:")
    start = st.checkbox("Enable webcam to register")

    class FaceRegister(VideoTransformerBase):
        def __init__(self):
            self.saved = False

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            boxes = detector.detect_faces(img)

            for (x1, y1, x2, y2) in boxes:
                face_crop = crop_face(img, (x1, y1, x2, y2))
                if face_crop.size == 0:
                    continue
                if not self.saved and name.strip():
                    emb = get_embedding(face_crop)
                    add_new_identity([emb], [name])
                    save_path = f"data/{name}"
                    os.makedirs(save_path, exist_ok=True)
                    cv2.imwrite(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", face_crop)
                    self.saved = True
                    print("Face registered")

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, "Saving..." if not self.saved else "Saved", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            return img

    if start and name.strip():
        webrtc_streamer(key="register", video_transformer_factory=FaceRegister)
    elif start:
        st.warning("Please enter a name before enabling webcam.")


# Webcam Face Recognition
elif tab == "Recognize Face":
    st.subheader("Real-time Face Recognition via Webcam")

    class FaceRecognize(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            boxes = detector.detect_faces(img)

            for (x1, y1, x2, y2) in boxes:
                face_crop = crop_face(img, (x1, y1, x2, y2))
                if face_crop.size == 0:
                    continue
                emb = get_embedding(face_crop)
                name, dist = find_match_faiss(emb)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{name} ({dist:.2f})" if name != "Unknown" else "Unknown"
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if name == "Unknown" else (0, 255, 0), 2)

            return img

    webrtc_streamer(key="recognize", video_transformer_factory=FaceRecognize)
