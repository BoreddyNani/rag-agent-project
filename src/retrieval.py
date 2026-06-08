# src/retrieval.py
from rank_bm25 import BM25Okapi
from ingestion import get_all_chunks  

# ── BM25 index ──────────────────────────────────────────
chunks = get_all_chunks()  
tokenized = [c.lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized)

def bm25_retrieve(query: str, n: int = 20) -> list[str]:
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:n]
    return [chunks[i] for i in top_idx]

def vector_retrieve(query: str, n: int = 20) -> list[str]:
    # your existing ChromaDB retrieval from ingestion.py
    from ingestion import retrieve
    return retrieve(query, n=n)

def rrf_fusion(bm25_ranks, vector_ranks, k=60) -> list:
    scores = {}
    for rank, doc in enumerate(bm25_ranks):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(vector_ranks):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

def hybrid_retrieve(query: str, n: int = 5) -> list[str]:
    bm25_results = bm25_retrieve(query, n=20)
    vector_results = vector_retrieve(query, n=20)
    fused = rrf_fusion(bm25_results, vector_results)
    return fused[:n]