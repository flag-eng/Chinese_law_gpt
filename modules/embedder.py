# modules/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name="shibing624/text2vec-base-chinese"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        emb = self.model.encode(texts, convert_to_numpy=True)
        return np.array(emb).astype("float32")
