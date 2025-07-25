# word2vec_utils.py
import os
import urllib.request
import gzip
import ssl
from gensim.models import KeyedVectors
import numpy as np


def load_word2vec_model():
    """
    Load Word2Vec model
    Returns the loaded KeyedVectors model.
    """
    download_dir = "downloads/word2vec"
    os.makedirs(download_dir, exist_ok=True)

    model_path = os.path.join(download_dir, "GoogleNews-vectors-negative300.bin.gz")

    if not os.path.exists(model_path):
        print("Word2Vec model not found.")
        return None
    try:
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


def create_concept_vectors(w2v_model, concepts):
    """
    Create vectors matrix for concepts using Word2Vec model.
    """
    if w2v_model is None:
        print("No model available")
        return None

    vectors = []
    missing_concepts = []

    for concept in concepts:
        concept_str = str(concept).strip().lower()

        variations = [
            concept_str,
            concept_str.capitalize(),
            concept_str.upper(),
            concept.strip() if isinstance(concept, str) else str(concept).strip()
        ]

        found = False
        for variation in variations:
            if variation in w2v_model:
                vectors.append(w2v_model[variation])
                found = True
                break

        if not found:
            # Instead of zeros, use random small vector to avoid zero norm issues
            random_vector = np.random.normal(0, 0.01, 300)
            vectors.append(random_vector)
            missing_concepts.append(concept)

    if missing_concepts:
        print(f"Warning: {len(missing_concepts)} concepts not found in model:")
        print(missing_concepts[:10])

    return np.array(vectors)