"""
chroma_retrieve.py — ChromaDB retriever for Traditional RAG lane.

Pipeline:
  1. Cosine similarity search — retrieve top 20 candidates
  2. Score threshold filter — drop anything below 0.25 similarity
  3. Cross-encoder rerank — same reranker as GraphRAG lane
  4. Return top_k after reranking
"""

import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from .config import settings
from .rerank import _load_cross_encoder

logger = logging.getLogger(__name__)

_CHROMA_PATH    = str(Path(__file__).parent.parent / "chroma_db")
_COLLECTION     = "bi_chunks"
_SCORE_THRESHOLD = 0.25   # drop chunks below this cosine similarity
_CANDIDATES      = 20     # how many to fetch before reranking


def _get_collection():
    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embed_model
    )
    return client.get_or_create_collection(
        name=_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


class ChromaRetriever:
    def __init__(self):
        self.collection = _get_collection()

    def retrieve(self, query: str, top_k: int = 5) -> dict:
        count = self.collection.count()
        if count == 0:
            return {"context": "", "citations": []}

        n = min(_CANDIDATES, count)
        results = self.collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        # Step 1: score threshold — drop clearly irrelevant chunks
        candidates = []
        for doc, meta, dist in zip(docs, metas, distances):
            score = 1 - dist
            if score >= _SCORE_THRESHOLD:
                candidates.append({"text": doc, "meta": meta, "cosine_score": score})

        if not candidates:
            return {"context": "", "citations": []}

        # Step 2: cross-encoder rerank and filter by rerank score
        encoder = _load_cross_encoder(settings.rerank_model)
        if encoder and len(candidates) > 1:
            pairs  = [(query, c["text"]) for c in candidates]
            scores = encoder.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
            # Keep only chunks the cross-encoder considers relevant (score > 0)
            candidates = [
                {**c, "rerank_score": float(s)}
                for c, s in ranked
                if float(s) > 0
            ]
            logger.info("ChromaDB cross-encoder: %d candidates → %d above threshold", len(ranked), len(candidates))

        candidates = candidates[:top_k]

        citations = []
        context_parts = []
        for c in candidates:
            citations.append({
                "text":        c["text"][:200],
                "source":      c["meta"].get("source", ""),
                "page_number": c["meta"].get("page_number") or None,
                "score":       round(c.get("rerank_score", c["cosine_score"]), 3),
            })
            context_parts.append(c["text"])

        context = "\n\n---\n\n".join(context_parts)
        return {"context": context, "citations": citations}
