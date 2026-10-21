
from functools import lru_cache

from tokenizers import Tokenizer

from config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNK_TOKENIZER_PATH


@lru_cache(maxsize=1)
def _load_tokenizer():
    if not CHUNK_TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            "Tokenizer Qwen introuvable : "
            f"{CHUNK_TOKENIZER_PATH}. Téléchargez tokenizer.json depuis "
            "Qwen/Qwen3.5-0.8B."
        )
    return Tokenizer.from_file(str(CHUNK_TOKENIZER_PATH))


def create_chunks(pages, chunk_size=None, chunk_overlap=None):
    """Split pages into Qwen-token windows while preserving source pages."""
    chunk_size = CHUNK_SIZE if chunk_size is None else chunk_size
    chunk_overlap = CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Il faut chunk_size > chunk_overlap >= 0.")

    tokenizer = _load_tokenizer()
    chunks = []
    step = chunk_size - chunk_overlap

    all_tokens = []

    for item in pages:
        page_tokens = tokenizer.encode(item["text"], add_special_tokens=False).ids
        all_tokens.extend((token, item["page"]) for token in page_tokens)

    for start in range(0, len(all_tokens), step):
        window = all_tokens[start:start + chunk_size]

        if not window:
            break

        token_ids = [token for token, _ in window]
        page_numbers = [page for _, page in window]
        text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()

        if text:
            chunks.append({
                "chunk_id": len(chunks),
                "text": text,
                "page_start": min(page_numbers),
                "page_end": max(page_numbers),
                "token_count": len(token_ids),
            })

        if start + chunk_size >= len(all_tokens):
            break

    print(
        f"{len(chunks)} chunks créés ({chunk_size} tokens, "
        f"chevauchement {chunk_overlap})."
    )
    return chunks
