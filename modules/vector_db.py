# modules/vector_db.py
import faiss
import numpy as np
import os
import json


class VectorDB:
    def __init__(self, dim, db_path="faiss.index", meta_path="meta.json"):
        self.dim = dim
        self.db_path = db_path
        self.meta_path = meta_path

        if os.path.exists(db_path):
            self.index = faiss.read_index(db_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.meta = []

    def add(self, embeddings, metadatas):
        self.index.add(embeddings)
        self.meta.extend(metadatas)

    def save(self):
        faiss.write_index(self.index, self.db_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def search(self, query_embedding, top_k=5):
        if len(self.meta) == 0:
            return []

        actual_top_k = min(top_k, len(self.meta))
        D, I = self.index.search(query_embedding, actual_top_k)
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.meta):
                results.append(self.meta[idx])
        return results
