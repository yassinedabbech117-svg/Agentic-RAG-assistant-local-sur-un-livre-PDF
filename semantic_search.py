"""Multilingual semantic retrieval powered by Ollama embeddings and FAISS."""

import faiss
import numpy as np
import requests

from config import (
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_EMBED_URL,
    SEMANTIC_INDEX_PATH,
    SEMANTIC_TOP_K,
)


def embed_texts(texts, batch_size=16):
    """Create embeddings in small batches to keep memory use manageable."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = []
    session = requests.Session()
    session.trust_env = False

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        print(
            f"Calcul des embeddings : passages "
            f"{start + 1}-{start + len(batch)} sur {len(texts)}...",
            flush=True,
        )

        try:
            response = session.post(
                OLLAMA_EMBED_URL,
                json={
                    "model": OLLAMA_EMBEDDING_MODEL,
                    "input": batch,
                    "truncate": True,
                    "keep_alive": "10m",
                },
                timeout=(5, 300),
            )
        except requests.exceptions.ConnectionError as error:
            raise RuntimeError(
                "Ollama n'est pas accessible pour créer les embeddings. "
                "Démarrez Ollama et réessayez."
            ) from error

        if response.status_code == 404:
            raise RuntimeError(
                f"Le modèle d'embeddings {OLLAMA_EMBEDDING_MODEL} est absent. "
                f"Téléchargez-le avec : ollama pull {OLLAMA_EMBEDDING_MODEL}"
            )

        response.raise_for_status()
        batch_embeddings = response.json().get("embeddings", [])

        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                "Ollama n'a pas retourné un embedding pour chaque passage."
            )

        embeddings.extend(batch_embeddings)

    return np.ascontiguousarray(embeddings, dtype=np.float32)


def build_semantic_index(chunks):
    """Embed all chunks and save a cosine-similarity FAISS index."""
    vectors = embed_texts([chunk["text"] for chunk in chunks])
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(SEMANTIC_INDEX_PATH))
    return index


def load_or_build_semantic_index(chunks):
    if SEMANTIC_INDEX_PATH.exists():
        return faiss.read_index(str(SEMANTIC_INDEX_PATH))

    return build_semantic_index(chunks)


def semantic_search(query, chunks, index, top_k=SEMANTIC_TOP_K):
    """Return chunk IDs ranked by multilingual cosine similarity."""
    vector = embed_texts([query])
    faiss.normalize_L2(vector)

    _, indices = index.search(vector, min(top_k, index.ntotal))
    return [int(idx) for idx in indices[0] if idx >= 0]
