# Face Detection & Recognition System

An end-to-end real-time Face Recognition system built using PyTorch, FAISS, Streamlit, and face detection models like MTCNN and MediaPipe.
Supports Face Registration, Real-Time Recognition, and Embedding Search with FAISS.

---

## Features

📸 Face registration via webcam (Streamlit or CLI)

🎯 Real-time face recognition using FAISS

🤖 Supports multiple face detectors (MTCNN, MediaPipe)

💾 Embedding extraction using Facenet (via facenet-pytorch)

⚡ Fast similarity search with FAISS

🖥️ Web-based UI using Streamlit

---

## Project Structure

    Face-Detection-and-Recognition/
    │
    ├── face_detectors/
    │   ├── detect_mtcnn.py
    │   ├── detect_mediapipe.py
    |   ├── detect_retinaface.py
    │
    ├── utils/
    │   ├── logger.py
    │   ├── embedder.py
    │   ├── matcher_faiss.py
    │   ├── preprocessing.py
    |   |__ matcher.py
    │
    ├── data/
    │   └── <saved_images_per_person>/
    │
    ├── embeddings/
    │   └── faiss.index
    │
    ├── register.py          # CLI-based registration
    ├── infer.py             # CLI-based recognition
    ├── app.py               # Streamlit App
    ├── requirements.txt
    └── README.md

---

## Installation

    # Create a virtual environment
    conda create -n <env-name> python=3.12 -y
    conda activate <env-name>

    # Install dependencies
    pip install -r requirements.txt

---

## Usage

    1. Register a Face (CLI)
    python register.py --name "<name>"

    2. Real-Time Inference (CLI)
    python infer.py    
    or
    python infer.py --video /path/to/video file

    3. Run Streamlit Web App
    streamlit run app.py

---

## Models Used

| Component         | Model / Library    |
| ----------------- | ------------------ |
| Face Detection    | MTCNN, MediaPipe   |
| Embedding Model   | facenet-pytorch    |
| Similarity Search | FAISS              |
| UI Framework      | Streamlit + WebRTC |
| Backend           | PyTorch, OpenCV    |

---
