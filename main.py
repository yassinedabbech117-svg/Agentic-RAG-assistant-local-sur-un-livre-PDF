
import json
import os

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNK_UNIT,
    CHUNKS_PATH,
    META_PATH,
    PDF_PATH,
    SEMANTIC_INDEX_PATH,
)

from hybrid_search import build_bm25, hybrid_search
from generator import generate_answer
from semantic_search import load_or_build_semantic_index


def initialize_rag():
    # Load PDF dependencies only when the first question is asked.
    from chunker import create_chunks
    from pdf_loader import load_pdf

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF introuvable : {PDF_PATH}")

    pdf_stat = PDF_PATH.stat()
    current_metadata = {
        "pdf_path": str(PDF_PATH.resolve()),
        "pdf_size": pdf_stat.st_size,
        "pdf_mtime_ns": pdf_stat.st_mtime_ns,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_unit": CHUNK_UNIT,
        "chunker_version": 2,
    }

    cached_chunks_are_current = False
    if CHUNKS_PATH.exists() and META_PATH.exists():
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                cached_metadata = json.load(f)
            cached_chunks_are_current = cached_metadata == current_metadata
        except (OSError, json.JSONDecodeError):
            cached_chunks_are_current = False

    if cached_chunks_are_current:
        try:
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if not isinstance(chunks, list) or not chunks:
                cached_chunks_are_current = False
        except (OSError, json.JSONDecodeError):
            cached_chunks_are_current = False

    if not cached_chunks_are_current:
        print(
            "Création des passages depuis le PDF "
            f"(taille {CHUNK_SIZE}, chevauchement {CHUNK_OVERLAP})..."
        )
        pages = load_pdf(PDF_PATH)
        chunks = create_chunks(pages)

        # An embedding index only matches the exact chunk list used to build it.
        if SEMANTIC_INDEX_PATH.exists():
            SEMANTIC_INDEX_PATH.unlink()

        CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        chunks_tmp = CHUNKS_PATH.with_suffix(CHUNKS_PATH.suffix + ".tmp")
        metadata_tmp = META_PATH.with_suffix(META_PATH.suffix + ".tmp")
        with open(chunks_tmp, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        with open(metadata_tmp, "w", encoding="utf-8") as f:
            json.dump(current_metadata, f, ensure_ascii=False, indent=2)
        os.replace(chunks_tmp, CHUNKS_PATH)
        os.replace(metadata_tmp, META_PATH)

    bm25 = build_bm25(chunks)
    semantic_index = load_or_build_semantic_index(chunks)

    return chunks, bm25, semantic_index


def main():
    print("Initialisation du RAG hybride...")

    chunks, bm25, semantic_index = initialize_rag()

    print("RAG prêt !")
    print("Écris 'exit' pour quitter.\n")

    while True:
        question = input("Question : ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            break

        if not question:
            continue

        try:
            results = hybrid_search(
                question,
                chunks,
                bm25,
                semantic_index,
            )

            if not results:
                print("Aucun passage pertinent trouvé.\n")
                continue

            print("\n--- Sources retrouvées ---")

            for chunk in results:
                print(
                    f"Chunk {chunk['chunk_id']} | "
                    f"pages {chunk['page_start']}-"
                    f"{chunk['page_end']}"
                )

            print("\n--- Réponse ---")
            print(generate_answer(question, results))
            print()

        except Exception as error:
            print(f"Une erreur est survenue : {error}")


if __name__ == "__main__":
    main()
