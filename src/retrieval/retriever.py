# src/retrieval/retriever.py

class Retriever:
    def __init__(self, vector_store, embedder, top_k=3):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query):
        query_vec = self.embedder.embed(query).astype("float32").reshape(1, -1)
        distances, indices = self.vector_store.index.search(query_vec, self.top_k)
        return indices[0], distances[0]
