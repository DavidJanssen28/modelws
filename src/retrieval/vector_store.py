import numpy as np
import faiss
import os
class VectorStore:
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.embeddings_path = "knowledge/processed/embeddings.npy"
        self.index_path = "knowledge/processed/index.faiss"
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
        else:
            self.embeddings = np.zeros((0, embedding_dim), dtype="float32")
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)
    def add(self, vector):
        vector = np.array(vector).astype("float32").reshape(1, -1)
        self.embeddings = np.vstack([self.embeddings, vector])
        self.index.add(vector)
        self.save()
    def save(self):
        np.save(self.embeddings_path, self.embeddings)
        faiss.write_index(self.index, self.index_path)
