import os
from rank_bm25 import BM25Okapi

# 1. Placeholders for global state that will be built dynamically on upload
bm25 = None
chunks_cache = []

def initialize_bm25_index():
    """
    Call this function dynamically inside your app.py or ingestion.py pipeline 
    RIGHT AFTER a new PDF is uploaded and processed into chunks.
    """
    global bm25, chunks_cache
    from ingestion import get_all_chunks
    
    # Fetch the newly extracted chunks
    chunks_cache = get_all_chunks()
    
    # DEFENSIVE GUARD: Avoid ZeroDivisionError if chunks are missing or empty
    if not chunks_cache:
        print("⚠️ Warning: Attempted to initialize BM25 with an empty corpus.")
        bm25 = None
        return
        
    # Build tokenized corpus safely
    tokenized = [c.lower().split() for c in chunks_cache]
    bm25 = BM25Okapi(tokenized)
    print(f"✅ BM25 Index successfully built with {len(chunks_cache)} chunks!")

def bm25_retrieve(query: str, n: int = 20) -> list[str]:
    global bm25, chunks_cache
    
    # COLD-START GUARD: Return empty list if no document has been uploaded yet
    if bm25 is None or not chunks_cache:
        return []
        
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(chunks_cache)), key=lambda i: scores[i], reverse=True)[:n]
    return [chunks_cache[i] for i in top_idx]

def vector_retrieve(query: str, n: int = 20) -> list[str]:
    # Defensive guard for ChromaDB if called before ingestion
    try:
        from ingestion import retrieve
        return retrieve(query, n=n)
    except Exception as e:
        print(f"Vector retrieval skipped or uninitialized: {e}")
        return []

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
    
    # If both pipelines are uninitialized, return an empty list gracefully
    if not bm25_results and not vector_results:
        return []
        
    fused = rrf_fusion(bm25_results, vector_results)
    return fused[:n]