import os
import cv2
import numpy as np
from tqdm import tqdm
import faiss
import pickle

from utils.logger import setup_logger
from face_detectors import detect_mtcnn
from feature_extractors.facenet_extractor import FaceNetEmbedder

logger = setup_logger()

def load_images_from_folder(folder_path):
    image_paths, labels = [], []
    for person in os.listdir(folder_path):
        person_dir = os.path.join(folder_path, person)
        if not os.path.isdir(person_dir): continue
        for img_name in os.listdir(person_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(person_dir, img_name))
                labels.append(person)
    return image_paths, labels

def build_faiss_index(data_dir="data", index_path="faiss_index/index.bin", label_path="faiss_index/labels.pkl"):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    embedder = FaceNetEmbedder()
    image_paths, labels = load_images_from_folder(data_dir)

    embeddings = []
    final_labels = []

    for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"Failed to load image: {img_path}")
            continue

        boxes = detect_mtcnn.detect_faces(image)
        if not boxes:
            logger.warning(f"No face detected: {img_path}")
            continue

        x1, y1, x2, y2 = boxes[0]
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            logger.warning(f"Invalid crop for: {img_path}")
            continue

        emb = embedder.get_embedding(face)
        embeddings.append(emb)
        final_labels.append(label)

    if not embeddings:
        logger.error("No valid embeddings found. Aborting.")
        return

    embeddings = np.array(embeddings).astype('float32')
    dim = embeddings.shape[1]

    logger.info(f"Building FAISS index with {len(embeddings)} vectors of dim {dim}")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(label_path, 'wb') as f:
        pickle.dump(final_labels, f)

    logger.info(f"Saved FAISS index to {index_path}")
    logger.info(f"Saved label mapping to {label_path}")

if __name__ == "__main__":
    build_faiss_index()
