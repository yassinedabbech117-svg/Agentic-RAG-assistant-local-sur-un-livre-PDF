"""End-to-end RAG evaluation: configurable chunks, hybrid top-5, Ollama 0.8B."""

import json
import argparse
import sys
import time
from datetime import datetime, timezone

import faiss

from chunker import create_chunks
from config import OLLAMA_EMBEDDING_MODEL, OLLAMA_MODEL, PDF_PATH
from generator import generate_answer
from hybrid_search import build_bm25, hybrid_search
from pdf_loader import load_pdf
from semantic_search import embed_texts


sys.stdout.reconfigure(encoding="utf-8", errors="replace")


TOP_K = 5
QUESTIONS = [
    {
        "id": 1,
        "question": "Why does the book say LLMs need external knowledge for RAG?",
        "expected_pages": [295],
        "expected_points": "RAG grounds model answers in an updatable external source; it addresses hallucination, stale knowledge, and lack of domain-specific knowledge.",
    },
    {
        "id": 2,
        "question": "How does the book distinguish the roles of RAG and fine-tuning?",
        "expected_pages": [296],
        "expected_points": "Fine-tuning teaches how to reason/respond; RAG supplies what to reason about. They complement one another.",
    },
    {
        "id": 3,
        "question": "What are the two main phases of a standard RAG pipeline?",
        "expected_pages": [296, 297],
        "expected_points": "Offline indexing processes and stores documents; online retrieval-generation finds relevant chunks and generates an answer.",
    },
    {
        "id": 4,
        "question": "When does sparse BM25 retrieval still have an advantage over dense retrieval?",
        "expected_pages": [297],
        "expected_points": "Exact lexical matching for product/error codes, proper nouns, rare terms, and low-resource domains; efficient without dense embedding search.",
    },
    {
        "id": 5,
        "question": "How does DPR learn to retrieve relevant passages?",
        "expected_pages": [298],
        "expected_points": "Separate query and passage encoders map relevant pairs close in embedding space using contrastive learning and in-batch negatives; hard negatives help.",
    },
    {
        "id": 6,
        "question": "Why does hybrid retrieval use reciprocal rank fusion instead of directly adding BM25 and cosine scores?",
        "expected_pages": [298, 561, 562],
        "expected_points": "BM25 and dense scores have different scales and are not directly comparable; RRF combines ranks, rewarding results ranked highly by both.",
    },
    {
        "id": 7,
        "question": "What does SPLADE combine, and how does it handle synonyms?",
        "expected_pages": [299, 300],
        "expected_points": "Sparse inverted-index retrieval with learned semantic term expansion; related terms can match despite different wording.",
    },
    {
        "id": 8,
        "question": "What is the purpose of overlap between fixed-size text chunks?",
        "expected_pages": [302],
        "expected_points": "Overlap preserves context across chunk boundaries so information split at a boundary remains available in neighboring chunks.",
    },
    {
        "id": 9,
        "question": "How does parent-child chunking work in the example?",
        "expected_pages": [303],
        "expected_points": "Small child chunks are retrieved for precision, while a larger parent chunk is returned to provide context.",
    },
    {
        "id": 10,
        "question": "What is a cross-encoder re-ranker used for, and why is it not the first-stage retriever?",
        "expected_pages": [304],
        "expected_points": "It jointly scores query-document pairs for more accurate ranking but costs more latency and cannot precompute document representations for first-stage search; use on a small candidate set.",
    },
    {
        "id": 11,
        "question": "What three retrieval modes does Adaptive RAG route queries to?",
        "expected_pages": [305],
        "expected_points": "No retrieval for simple queries, single-step RAG for moderate queries, and multi-step RAG for complex multi-hop questions.",
    },
    {
        "id": 12,
        "question": "What kind of questions is Graph RAG especially suited to answer?",
        "expected_pages": [305],
        "expected_points": "Global questions requiring synthesis across a corpus, using extracted entities/relationships, communities, and hierarchical summaries; it is more expensive to build and maintain.",
    },
    {
        "id": 13,
        "question": "In the book's MDP framing of Agentic RAG, what are the state, actions, and reward?",
        "expected_pages": [307],
        "expected_points": "State is the query plus retrieved documents/current context; actions include retrieve, reason, generate, stop; reward is answer correctness.",
    },
    {
        "id": 14,
        "question": "What are the three multi-source routing strategies and one trade-off of each?",
        "expected_pages": [307, 308],
        "expected_points": "Rule-based (fast/interpretable but brittle), classifier-based (low latency but needs labeled data), LLM-based (flexible but adds an LLM call/latency).",
    },
    {
        "id": 15,
        "question": "What four stages make up the full Agentic RAG implementation described in the book?",
        "expected_pages": [309, 310],
        "expected_points": "Plan/decompose, retrieve, evaluate whether context is sufficient, and generate; insufficient context loops to retrieval until sufficient or budget exhausted.",
    },
    {
        "id": 16,
        "question": "What makes Search-R1 an RL-trained Agentic RAG method?",
        "expected_pages": [311, 312],
        "expected_points": "The model learns when/how to search through interaction/retrieval actions and receives reward for answer quality, instead of relying only on fixed retrieval before generation.",
    },
    {
        "id": 17,
        "question": "Why do agents need memory according to the start of Chapter 17?",
        "expected_pages": [320],
        "expected_points": "LLMs are stateless; memory preserves useful information across interactions/tasks beyond the current prompt/context.",
    },
    {
        "id": 18,
        "question": "Which components compete for an agent's context-window token budget?",
        "expected_pages": [344],
        "expected_points": "System prompt, memory/RAG, tool definitions, conversation history, and reserved output tokens.",
    },
    {
        "id": 19,
        "question": "Name two methods the book gives for compressing or retaining context when history grows too large.",
        "expected_pages": [345, 346],
        "expected_points": "Examples include summarizing old turns, selective retention by query relevance/recency, importance-weighted truncation, or sliding-window policies such as FIFO and importance-ranked retention.",
    },
    {
        "id": 20,
        "question": "Describe the ReAct cycle and two implementation safeguards mentioned in the book.",
        "expected_pages": [371],
        "expected_points": "Thought/reasoning, action/tool call, observation loop until final answer; safeguards include hidden scratchpad, structured tool parsing, max-iteration cap, and explicit termination.",
    },
]


def build_cosine_index(chunks, cache_path):
    if cache_path.exists():
        cached = faiss.read_index(str(cache_path))
        if cached.ntotal == len(chunks):
            print(f"Index sémantique réutilisé : {cached.ntotal} vecteurs.", flush=True)
            return cached

    vectors = embed_texts([chunk["text"] for chunk in chunks])
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(cache_path))
    return index


def relevant(chunk, expected_pages):
    return any(chunk["page_start"] <= page <= chunk["page_end"] for page in expected_pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()
    chunk_size, chunk_overlap = args.chunk_size, args.overlap
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        parser.error("L'overlap doit être positif ou nul et inférieur au chunk size.")

    pages = load_pdf(PDF_PATH)
    chunks = create_chunks(pages, chunk_size, chunk_overlap)
    print(f"Création BM25 sur {len(chunks)} chunks...", flush=True)
    bm25 = build_bm25(chunks)
    output_path = PDF_PATH.parent / "rag_data" / f"rag_20_questions_{chunk_size}_{chunk_overlap}.json"
    index_path = PDF_PATH.parent / "rag_data" / f"eval-{chunk_size}-{chunk_overlap}-qwen-embedding.faiss"
    if output_path.exists():
        report = json.loads(output_path.read_text(encoding="utf-8"))
        if report.get("chunk_count") != len(chunks) or report.get("top_k") != TOP_K:
            raise RuntimeError("Le rapport existant ne correspond pas à cette configuration.")
    else:
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunk_size_tokens": chunk_size,
            "overlap_tokens": chunk_overlap,
            "chunk_count": len(chunks),
            "retriever": "BM25 + cosine semantic search + reciprocal rank fusion",
            "top_k": TOP_K,
            "embedding_model": OLLAMA_EMBEDDING_MODEL,
            "generator_model": OLLAMA_MODEL,
            "questions": [],
        }

    print("Création de l'index sémantique...", flush=True)
    index = build_cosine_index(chunks, index_path)
    completed_ids = {row["id"] for row in report["questions"]}

    for number, item in enumerate(QUESTIONS, start=1):
        if item["id"] in completed_ids:
            print(f"\n[{number}/{len(QUESTIONS)}] déjà enregistrée; reprise...", flush=True)
            continue
        print(f"\n[{number}/{len(QUESTIONS)}] {item['question']}", flush=True)
        started = time.perf_counter()
        retrieved = hybrid_search(
            item["question"], chunks, bm25, index, top_k=TOP_K
        )
        retrieval_seconds = time.perf_counter() - started
        relevant_ranks = [
            rank for rank, chunk in enumerate(retrieved, 1)
            if relevant(chunk, item["expected_pages"])
        ]

        started = time.perf_counter()
        answer = generate_answer(
            item["question"], retrieved, model_name=OLLAMA_MODEL, stream=False
        )
        generation_seconds = time.perf_counter() - started

        result = {
            **item,
            "retrieval": {
                "seconds": round(retrieval_seconds, 2),
                "hit_at_5": bool(relevant_ranks),
                "first_relevant_rank": relevant_ranks[0] if relevant_ranks else None,
                "relevant_count_at_5": len(relevant_ranks),
                "sources": [
                    {
                        "rank": rank,
                        "pages": [chunk["page_start"], chunk["page_end"]],
                        "relevant_by_page": relevant(chunk, item["expected_pages"]),
                        "excerpt": chunk["text"][:400],
                    }
                    for rank, chunk in enumerate(retrieved, start=1)
                ],
            },
            "generation": {
                "seconds": round(generation_seconds, 2),
                "answer": answer,
            },
        }
        report["questions"].append(result)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"Sources={[ (c['page_start'], c['page_end']) for c in retrieved ]}; "
            f"retrieval={retrieval_seconds:.1f}s; generation={generation_seconds:.1f}s",
            flush=True,
        )
        print(answer[:600].replace("\n", " "), flush=True)

    results = report["questions"]
    retrieval_hits = sum(q["retrieval"]["hit_at_5"] for q in results)
    report["summary"] = {
        "retrieval_hit_at_5": f"{retrieval_hits}/{len(results)}",
        "mean_retrieval_seconds": round(
            sum(q["retrieval"]["seconds"] for q in results) / len(results), 2
        ),
        "mean_generation_seconds": round(
            sum(q["generation"]["seconds"] for q in results) / len(results), 2
        ),
    }
    if all("faithfulness_score_0_to_2" in q["generation"] for q in results):
        score_sum = sum(q["generation"]["faithfulness_score_0_to_2"] for q in results)
        report["summary"]["manual_faithfulness_score_out_of_10"] = round(
            score_sum / (2 * len(results)) * 10, 2
        )
        report["summary"]["manual_judgment_scale"] = (
            "0=wrong/unsupported, 1=partly correct with material issues, "
            "2=correct and grounded; scores scaled to 10."
        )
    else:
        report["summary"]["manual_faithfulness_scores_pending"] = True
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRapport enregistré : {output_path}", flush=True)


if __name__ == "__main__":
    main()
