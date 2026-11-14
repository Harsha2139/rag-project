import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):
        self.dimension = dimension

        # Create FAISS index
        self.index = faiss.IndexFlatL2(dimension)

        self.chunks = []

    def add_embeddings(self, embeddings, chunks):

        embeddings = np.array(
            embeddings,
            dtype="float32"
        )

        self.index.add(embeddings)

        self.chunks.extend(chunks)

        print("Vectors added successfully!")

    def search(self, query_embedding, top_k=2):

        query_embedding = np.array(
            [query_embedding],
            dtype="float32"
        )

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for index in indices[0]:

            if index != -1:
                results.append(self.chunks[index])

        return results


if __name__ == "__main__":

    sample_chunks = [
        "Python is a programming language.",
        "Loops repeat a block of code.",
        "Functions organize reusable code."
    ]

    embeddings = np.array([
        [0.1, 0.2, 0.3],
        [0.8, 0.7, 0.6],
        [0.2, 0.3, 0.4]
    ], dtype="float32")

    store = VectorStore(dimension=3)

    store.add_embeddings(
        embeddings,
        sample_chunks
    )

    query = np.array([0.15, 0.25, 0.35])

    results = store.search(query, top_k=2)

    print("\nSearch Results:")

    for result in results:
        print("-", result)