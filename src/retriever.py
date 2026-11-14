
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.document_loader import load_pdf
from src.text_chunker import create_chunks
from src.vector_store import VectorStore


# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def build_search_system(pdf_paths):

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

                if not chunk.strip():
                    continue

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

    # Generate normalized embeddings
    embeddings = model.encode(
        all_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    store = VectorStore(embeddings.shape[1])

    store.add_embeddings(
        embeddings,
        all_chunks
    )

    return store, metadata


def retrieve(question, store, metadata, top_k=5):

    if not question.strip():
        return []

    if store is None or not metadata:
        return []

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    results = store.search(
        question_embedding[0],
        top_k=top_k
    )

    relevant_results = []

    # Initial squared L2 distance threshold.
    # Lower distance means greater similarity.
    DISTANCE_THRESHOLD = 1.0

    for result in results:

        idx = result["index"]
        distance = result["distance"]

        if idx >= len(metadata):
            continue

        if distance > DISTANCE_THRESHOLD:
            continue

        item = metadata[idx].copy()

        item["distance"] = distance
        item["similarity"] = 1 - (distance / 2)

        relevant_results.append(item)

    return relevant_results
