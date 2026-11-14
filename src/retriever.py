
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.document_loader import load_pdf
from src.text_chunker import create_chunks
from src.vector_store import VectorStore


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

DISTANCE_THRESHOLD = 1.0


# =========================================================
# BUILD KNOWLEDGE BASE
# =========================================================

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


# =========================================================
# MULTI-DOCUMENT RETRIEVAL
# =========================================================

def retrieve(
    question,
    store,
    metadata,
    top_k=5,
    distance_threshold=DISTANCE_THRESHOLD
):

    if not question or not question.strip():
        return []

    if store is None or not metadata:
        return []

    # Generate question embedding
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query = np.asarray(
        question_embedding,
        dtype="float32"
    )

    # Search the complete FAISS index
    total_vectors = store.index.ntotal

    if total_vectors == 0:
        return []

    distances, indices = store.index.search(
        query,
        total_vectors
    )

    # Maximum relevant chunks from each PDF
    MAX_CHUNKS_PER_DOCUMENT = max(
        1,
        min(int(top_k), 5)
    )

    document_counts = {}
    relevant_results = []

    # Process results from most relevant to least relevant
    candidates = sorted(
        zip(distances[0], indices[0]),
        key=lambda item: item[0]
    )

    for distance, idx in candidates:

        if idx == -1 or idx >= len(metadata):
            continue

        # Ignore weak matches
        if distance > distance_threshold:
            continue

        item = metadata[idx]

        filename = item.get(
            "file",
            "Unknown PDF"
        )

        # Balance retrieval across documents
        current_count = document_counts.get(
            filename,
            0
        )

        if current_count >= MAX_CHUNKS_PER_DOCUMENT:
            continue

        result = item.copy()

        result["distance"] = float(distance)
        result["similarity"] = float(
            1 - (distance / 2)
        )

        relevant_results.append(result)

        document_counts[filename] = current_count + 1

    return relevant_results


# =========================================================
# GET AVAILABLE DOCUMENTS
# =========================================================

def get_document_names(metadata):

    return sorted(set(
        item.get("file", "Unknown PDF")
        for item in metadata
    ))


# =========================================================
# GET ALL CONTENT FROM ONE DOCUMENT
# =========================================================

def get_document_chunks(metadata, filename):

    return [
        item
        for item in metadata
        if item.get("file") == filename
    ]


# =========================================================
# GET ALL INDEXED DOCUMENT CONTENT
# =========================================================

def get_all_document_chunks(metadata):

    grouped_documents = {}

    for item in metadata:

        filename = item.get(
            "file",
            "Unknown PDF"
        )

        if filename not in grouped_documents:
            grouped_documents[filename] = []

        grouped_documents[filename].append(item)

    return grouped_documents
