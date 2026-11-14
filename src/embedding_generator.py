from sentence_transformers import SentenceTransformer


# Load a pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks):

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    return embeddings


if __name__ == "__main__":

    sample_chunks = [
        "Python is a programming language.",
        "Loops repeat a block of code.",
        "Functions organize reusable code."
    ]

    vectors = generate_embeddings(sample_chunks)

    print("Embeddings generated successfully!")

    print("Number of chunks:", len(sample_chunks))
    print("Vector shape:", vectors.shape)

    print("\nFirst vector:")
    print(vectors[0])