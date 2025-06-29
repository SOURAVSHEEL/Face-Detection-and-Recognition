# Face Recognition & Anti-Spoofing System

Real-time face recognition system with anti-spoofing using Deep Learning, FAISS search, and  OpenCV. It combines detection, recognition, Anti-Spoofing 

![Demo](assets/inference_output.gif)

---

## 🚀 Features

- **Multiple Face Detection Models**: Support for MediaPipe, MTCNN, and RetinaFace
- **Anti-Spoofing Detection**: DeepPixBiS implementation for liveness detection
- **Face Recognition**: FaceNet-based feature extraction and similarity matching
- **Real-time Processing**: Live face detection and recognition capabilities
- **FAISS Integration**: Efficient similarity search using Facebook AI Similarity Search
- **Comprehensive Training Pipeline**: End-to-end training workflow for custom models

---

## 📊 Models

### Face Detection

- **MediaPipe**: Fast and lightweight face detection
- **MTCNN**: Multi-task CNN for face detection and alignment
- **RetinaFace**: High-accuracy face detection with landmarks

### Anti-Spoofing

- **DeepPixBiS**: Deep Pixel-wise Binary Supervision for face anti-spoofing
  - Multiple model versions (v1, v2, v3) available, best version is v3 
  - Trained on CASIA-FASD (Face Anti-Spoofing Database) dataset
  - Performs pixel-wise binary classification for liveness detection

### Face Recognition

- **FaceNet**: Deep learning model for face verification and recognition
- **Feature Extraction**: 512-dimensional face embeddings
- **FAISS Indexing**: Fast similarity search for large face databases

---

## 🎯 Usage

### Training

1. **Prepare your dataset**

   ```bash
   src/data_preparation.py
   ```

2. **DeepPixBis  implementation**

   ```bash
   src/deepPixBis_model.py
   ```

3. **Training loop**

   ```bash
   src/train.py
   ```

4. **Training Pipeline**

   ```bash
   python src/train_pipeline.py
   ```

### Testing

1. **Real-time testing**

   ```bash
   python src/realtime_test.py
   ```

### Face Registration

1. **Register new faces**

   ```bash
   python register.py  --name "Name"
   ```

2. **Build face embeddings**

   ```bash
   python feature_extractors/build_embeddings.py
   ```

### Inference

```bash
python feature_extractors/infer.py
```

---

## 📊 Dataset Information

### CASIA-FASD Dataset

This project uses the **CASIA Face Anti-Spoofing Database** for training the DeepPixBiS model:

- **Attack Types**: 
  - Warped photo attacks
  - Cut photo attacks  
  - Video replay attacks
- **Variations**: Different lighting conditions, backgrounds, and camera qualities
- **Labels**: Binary classification (live vs. spoof)
- **Training/Testing Split**: Organized in `dataset/train/` and `dataset/test/` directories

## 🗂️ Dataset Structure

Organize your dataset as follows:

dataset/
├── train/
│   ├── live/     # Real face images
│   └── spoof/    # Spoofed/fake face images
└── test/
    ├── live/     # Test real face images
    └── spoof/    # Test spoofed face images

---

## DeepPixBiS Test Results (CASIA-FASD)

- **Overall Accuracy**: 96.91%
- **F1 Score**: 96.85%
- **Decision Threshold**: 0.50

## Class-wise Performance:

**Spoof Detection**:

- **Precision**: 95%
- **Recall**: 99%
- **F1-Score**: 97%

## Live Detection:

- **Precision**: 99%
- **Recall**: 95%
- **F1-Score**: 97%

## Error Analysis:

- **False Positives**: 98 (Live faces classified as Spoof)
- **False Negatives**: 527 (Spoof faces classified as Live)

---

## 🙏 Acknowledgments

- **CASIA-FASD Dataset**: Chinese Academy of Sciences' Institute of Automation for providing the Face Anti-Spoofing Database
- **DeepPixBiS**: Original paper and implementation for pixel-wise binary supervision
- **FaceNet**: For face recognition capabilities
- **MediaPipe, MTCNN, and RetinaFace**: For face detection algorithms
- **FAISS**: For efficient similarity search implementation

---
