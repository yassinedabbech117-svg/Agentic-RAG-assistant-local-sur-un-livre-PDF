"""Compare BM25, semantic, and hybrid retrieval across chunk sizes."""

import json
import time
from datetime import datetime, timezone

import faiss

from chunker import create_chunks
from config import OLLAMA_EMBEDDING_MODEL, PDF_PATH
from hybrid_search import build_bm25, lexical_search, reciprocal_rank_fusion
from pdf_loader import load_pdf
from semantic_search import embed_texts, semantic_search


CHUNK_CONFIGS = [(300, 50), (600, 120)]
TOP_K = 5
QUESTIONS = [
    {
        "id": "routing",
        "question": "What are the three multi-source routing strategies described in the book?",
        "expected_pages": [306, 307, 308, 309],
    },
    {
        "id": "rag_finetuning",
        "question": "How do RAG and fine-tuning complement each other?",
        "expected_pages": [316, 317, 318],
    },
    {
        "id": "react_cycle",
        "question": "What does a ReAct agent do in its Thought-Action-Observation cycle?",
        "expected_pages": [352, 353, 354, 355],
    },
    {
        "id": "unanswerable",
        "question": "What is my personal bank account balance?",
        "expected_pages": [],
    },
]


def build_cosine_index(chunks):
    vectors = embed_texts([chunk["text"] for chunk in chunks])
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def is_relevant(chunk, expected_pages):
    return any(
        chunk["page_start"] <= page <= chunk["page_end"]
        for page in expected_pages
    )


def score_ranking(ranked_chunks, expected_pages):
    if not expected_pages:
        return {"hit_at_5": None, "mrr": None, "precision_at_5": None}

    relevant_ranks = [
        rank
        for rank, chunk in enumerate(ranked_chunks, start=1)
        if is_relevant(chunk, expected_pages)
    ]
    return {
        "hit_at_5": bool(relevant_ranks),
        "mrr": 1 / relevant_ranks[0] if relevant_ranks else 0.0,
        "precision_at_5": len(relevant_ranks) / TOP_K,
    }


def describe_chunks(ranked_chunks, expected_pages):
    return [
        {
            "rank": rank,
            "pages": [chunk["page_start"], chunk["page_end"]],
            "relevant_by_page": (
                is_relevant(chunk, expected_pages) if expected_pages else None
            ),
            "excerpt": chunk["text"][:280],
        }
        for rank, chunk in enumerate(ranked_chunks, start=1)
    ]


def evaluate_configuration(pages, chunk_size, overlap):
    started = time.perf_counter()
    chunks = create_chunks(pages, chunk_size=chunk_size, chunk_overlap=overlap)
    chunking_seconds = time.perf_counter() - started

    started = time.perf_counter()
    bm25 = build_bm25(chunks)
    bm25_build_seconds = time.perf_counter() - started

    started = time.perf_counter()
    index = build_cosine_index(chunks)
    semantic_build_seconds = time.perf_counter() - started

    results = []
    for item in QUESTIONS:
        query = item["question"]
        expected_pages = item["expected_pages"]
        timings = {}

        started = time.perf_counter()
        bm25_ids = lexical_search(query, bm25, top_k=TOP_K)
        timings["bm25_seconds"] = round(time.perf_counter() - started, 3)

        started = time.perf_counter()
        semantic_ids = semantic_search(query, chunks, index, top_k=TOP_K)
        timings["semantic_seconds"] = round(time.perf_counter() - started, 3)

        rankings = {
            "BM25": [chunks[i] for i in bm25_ids],
            "Semantic cosine": [chunks[i] for i in semantic_ids],
            "Hybrid RRF": [
                chunks[i]
                for i in reciprocal_rank_fusion(
                    [semantic_ids, bm25_ids], top_k=TOP_K
                )
            ],
        }
        retrievers = {}
        for name, ranked_chunks in rankings.items():
            retrievers[name] = {
                **score_ranking(ranked_chunks, expected_pages),
                "results": describe_chunks(ranked_chunks, expected_pages),
            }

        results.append({
            "id": item["id"],
            "question": query,
            "expected_pages": expected_pages,
            "timings": timings,
            "retrievers": retrievers,
        })

    answerable = [row for row in results if row["expected_pages"]]
    summary = {}
    for name in ("BM25", "Semantic cosine", "Hybrid RRF"):
        rows = [row["retrievers"][name] for row in answerable]
        hit_rate = sum(row["hit_at_5"] for row in rows) / len(rows)
        mean_mrr = sum(row["mrr"] for row in rows) / len(rows)
        mean_precision = sum(row["precision_at_5"] for row in rows) / len(rows)
        # Transparent weighted 0-10 diagnostic score; rankings remain available
        # so the user can inspect the result behind each metric.
        rating = 10 * (
            0.5 * hit_rate + 0.3 * mean_mrr + 0.2 * mean_precision
        )
        summary[name] = {
            "hit_rate_at_5": round(hit_rate, 3),
            "mean_reciprocal_rank": round(mean_mrr, 3),
            "mean_precision_at_5": round(mean_precision, 3),
            "rating_out_of_10": round(rating, 2),
        }

    return {
        "chunk_size_tokens": chunk_size,
        "overlap_tokens": overlap,
        "chunk_count": len(chunks),
        "chunking_seconds": round(chunking_seconds, 2),
        "bm25_build_seconds": round(bm25_build_seconds, 2),
        "semantic_index_build_seconds": round(semantic_build_seconds, 2),
        "summary": summary,
        "questions": results,
    }


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF introuvable : {PDF_PATH}")

    pages = load_pdf(PDF_PATH)
    configurations = []
    for chunk_size, overlap in CHUNK_CONFIGS:
        print(f"\nÉvaluation chunks={chunk_size}, overlap={overlap}", flush=True)
        configurations.append(
            evaluate_configuration(pages, chunk_size, overlap)
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pdf": PDF_PATH.name,
        "tokenizer": "Qwen/Qwen3.5-0.8B tokenizer",
        "embedding_model": OLLAMA_EMBEDDING_MODEL,
        "top_k": TOP_K,
        "scoring": (
            "On 3 answerable questions: rating = 10 * "
            "(0.5*Hit@5 + 0.3*MRR + 0.2*Precision@5). "
            "Gold page overlap is the relevance proxy. The unanswerable query "
            "is reported separately and excluded from the rating."
        ),
        "configurations": configurations,
    }
    output_path = PDF_PATH.parent / "rag_data" / "retriever_chunk_comparison.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRapport enregistré : {output_path}")
    for config in configurations:
        print(
            f"\n{config['chunk_size_tokens']}/{config['overlap_tokens']} "
            f"({config['chunk_count']} chunks)"
        )
        for name, metrics in config["summary"].items():
            print(
                f"{name}: {metrics['rating_out_of_10']}/10 "
                f"(Hit@5={metrics['hit_rate_at_5']}, "
                f"MRR={metrics['mean_reciprocal_rank']}, "
                f"P@5={metrics['mean_precision_at_5']})"
            )


if __name__ == "__main__":
    main()
