
import re

from rank_bm25 import BM25Okapi

from config import BM25_TOP_K, FINAL_TOP_K
from semantic_search import semantic_search as find_semantic_matches


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25(chunks):
    corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus)


def lexical_search(query, bm25, top_k=BM25_TOP_K):
    scores = bm25.get_scores(tokenize(query))
    ranked = scores.argsort()[::-1][:top_k]

    return [
        int(idx)
        for idx in ranked
        # BM25 scores can be negative for common terms; any non-zero score
        # still means the passage matched at least one query token.
        if scores[idx] != 0
    ]


def reciprocal_rank_fusion(rankings, top_k=FINAL_TOP_K, k=60):
    fused = {}

    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = (
                fused.get(chunk_id, 0.0)
                + 1.0 / (k + rank)
            )

    return sorted(
        fused,
        key=fused.get,
        reverse=True,
    )[:top_k]


def hybrid_search(
    query,
    chunks,
    bm25,
    semantic_index,
    top_k=FINAL_TOP_K,
):
    lexical_ids = lexical_search(query, bm25)
    semantic_ids = find_semantic_matches(
        query,
        chunks,
        semantic_index,
    )

    final_ids = reciprocal_rank_fusion(
        [semantic_ids, lexical_ids],
        top_k=top_k,
    )

    return [chunks[idx] for idx in final_ids]
