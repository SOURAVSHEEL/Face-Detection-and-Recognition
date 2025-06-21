import faiss
import numpy as np
import os
import pickle

# Paths
INDEX_PATH = "embeddings/faiss.index"
META_PATH = "embeddings/meta.pkl"

# Load FAISS index
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexFlatL2(512)  # 512-dim FaceNet vectors

# Load metadata (names list)
if os.path.exists(META_PATH):
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    name_list = metadata["names"]
else:
    name_list = []
    metadata = {"names": []}
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)


def find_match_faiss(query_emb, threshold=1.0):
    """
    Perform nearest neighbor search using FAISS.
    
    Args:
        query_emb (np.ndarray): shape (512,) — input embedding
        threshold (float): maximum allowed L2 distance for valid match

    Returns:
        name (str), distance (float)
    """
    query_emb = np.array([query_emb]).astype("float32")
    D, I = index.search(query_emb, k=1)  # top 1 match
    dist, idx = D[0][0], I[0][0]

    if idx == -1 or dist > threshold:
        return "Unknown", dist
    return name_list[idx], dist


def add_new_identity(embeddings, names):
    """
    Add new embeddings and names to the FAISS index and save metadata.

    Args:
        embeddings (List[np.ndarray]): list of 512-dim vectors
        names (List[str]): corresponding person names
    """
    global name_list, index

    embeddings = np.array(embeddings).astype("float32")
    index.add(embeddings)
    name_list.extend(names)

    # Save updated index and metadata
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump({"names": name_list}, f)

    print(f"✅ Added {len(names)} identities to FAISS index.")
