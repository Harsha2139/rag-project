from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.document_loader import load_pdf
from src.text_chunker import create_chunks
from src.vector_store import VectorStore


# Load the embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def build_search_system(pdf_paths):
    """
    Read multiple PDFs, split their text into chunks,
    generate embeddings and store them in FAISS.
    """

    all_chunks = []
    metadata = []

    for pdf_path in pdf_paths:
        pages = load_pdf(pdf_path)

        for page in pages:
            page_text = page.get("text", "").strip()

            if not page_text:
                continue

            chunks = create_chunks(
                page_text,
                chunk_size=500,
                overlap=50
            )

            for chunk in chunks:
                all_chunks.append(chunk)

                metadata.append({
                    "file": Path(pdf_path).name,
                    "page": page["page"],
                    "text": chunk
                })

    if not all_chunks:
        raise ValueError(
            "No readable text was found in the selected PDFs."
        )

    embeddings = model.encode(
        all_chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    store = VectorStore(embeddings.shape[1])
    store.add_embeddings(embeddings, all_chunks)

    return store, metadata


def retrieve(question, store, metadata, top_k=5):
    """Find the most relevant chunks for a question."""

    if store is None or not metadata:
        return []

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    query = np.asarray(question_embedding, dtype="float32")

    number_of_results = min(top_k, store.index.ntotal)

    if number_of_results == 0:
        return []

    distances, indices = store.index.search(
        query,
        number_of_results
    )

    results = []

    for idx in indices[0]:
        if idx != -1 and idx < len(metadata):
            results.append(metadata[idx])

    return results