
from main import initialize_rag
from hybrid_search import hybrid_search
from generator import generate_answer


def initialize_service():
    """Charge ou construit les composants du moteur RAG."""
    return initialize_rag()


def ask_question(question, rag_state, top_k=5, model_name=None, stream=False):
    """Recherche les passages pertinents et génère une réponse."""

    chunks, bm25, semantic_index = rag_state

    results = hybrid_search(
        question,
        chunks,
        bm25,
        semantic_index,
        top_k=top_k,
    )

    if not results:
        return {
            "answer": "Aucun passage pertinent trouvé dans le livre.",
            "sources": [],
        }

    answer = generate_answer(
        question,
        results,
        model_name=model_name,
        stream=stream,
    )

    sources = [
        {
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "text": chunk.get("text", ""),
            "chunk_id": chunk.get("chunk_id"),
        }
        for chunk in results
    ]

    return {
        "answer": answer,
        "sources": sources,
    }
