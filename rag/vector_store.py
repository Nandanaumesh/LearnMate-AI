import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):
        """
        Create a FAISS index.

        dimension:
            Number of values in each embedding vector.
        """

        self.dimension = dimension

        self.index = faiss.IndexFlatL2(dimension)

        self.chunks = []


    def add_embeddings(self, embeddings, chunks):
        """
        Add embeddings and their corresponding chunks
        to the FAISS index.
        """

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.index.add(embeddings)

        self.chunks.extend(chunks)


    def search(self, query_embedding, top_k=3):
        """
        Search for the most similar chunks.
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        query_embedding = query_embedding.reshape(
            1,
            -1
        )

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            if index == -1:
                continue

            results.append({
                "chunk": self.chunks[index],
                "distance": float(distance)
            })

        return results