
import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []

    def add_embeddings(self, embeddings, chunks):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch.")

        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings and chunks must match.")

        self.index.add(embeddings)
        self.chunks.extend(chunks)

        print("Vectors added successfully!")

    def search(self, query_embedding, top_k=5):

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError("Query embedding dimension mismatch.")

        top_k = min(top_k, self.index.ntotal)

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for distance, index in zip(distances[0], indices[0]):

            if index == -1:
                continue

            results.append({
                "index": int(index),
                "text": self.chunks[index],
                "distance": float(distance)
            })

        return results
