
def create_chunks(text, chunk_size=500, overlap=50):
    """
    Split text into smaller overlapping chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    sample_text = """
    Python is a programming language.
    Python supports loops, functions and classes.
    Loops are used to repeat statements.
    """

    result = create_chunks(sample_text, chunk_size=50, overlap=10)

    for i, chunk in enumerate(result, start=1):
        print(f"\nChunk {i}:")
        print(chunk)

    print("\nTotal chunks:", len(result))
