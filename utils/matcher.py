import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load saved embeddings and names
with open("embeddings/meta.pkl", "rb") as f:
    metadata = pickle.load(f)  # {'embeddings': [...], 'names': [...]}

registered_embeddings = np.array(metadata['embeddings'])
registered_names = metadata['names']

def find_match(query_embedding, threshold=0.6):
    """
    Match face embedding with stored ones using cosine similarity.
    Args:
        query_embedding (np.ndarray): 512-dim vector
        threshold (float): Minimum similarity to be considered a match
    Returns:
        name (str), similarity (float)
    """
    sims = cosine_similarity([query_embedding], registered_embeddings)[0]
    max_idx = np.argmax(sims)
    max_sim = sims[max_idx]

    if max_sim > threshold:
        return registered_names[max_idx], max_sim
    else:
        return "Unknown", max_sim
