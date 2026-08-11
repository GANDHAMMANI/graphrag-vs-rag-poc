"""
rerank.py — Two-stage relevance ranking for GraphRAG retrieval
--------------------------------------------------------------
Stage 1 — RRF (Reciprocal Rank Fusion)
  Fuses keyword-match and semantic-match seed entity rankings into one
  ordered list. Items that appear in both lists score higher than items
  that appear in only one.

Stage 2 — Cross-encoder reranker
  Scores each (query, fact_text) pair with a local bi-encoder model
  (ms-marco-MiniLM-L-6-v2). Much cheaper than a full LLM reranker and
  better than cosine similarity for relevance ranking because it sees
  the query and the fact together
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# RRF smoothing constant. k=60 is the standard value from the original paper.
# Higher k → less penalty for lower-ranked items.
_RRF_K = 60

# Lazy-loaded singleton so the model loads once and is reused across requests.
_cross_encoder = None


# ── Stage 1: RRF ──────────────────────────────────────────────────────────────

def rrf_merge(ranked_lists: list[list[str]], k: int = _RRF_K) -> list[str]:
    """
    Reciprocal Rank Fusion over multiple ranked lists of entity names.

    Score per entity = Σ  1 / (k + rank_in_list_i)
    Items present in multiple lists benefit from score accumulation —
    exactly what we want when both keyword and semantic search agree.

    Parameters
    ----------
    ranked_lists : each inner list is already ordered best-first
    k            : RRF smoothing constant (default 60)

    Returns
    -------
    Single merged list, best-first by fused score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, name in enumerate(ranked, start=1):
            scores[name] = scores.get(name, 0.0) + 1.0 / (k + rank)

    return sorted(scores, key=scores.__getitem__, reverse=True)


# ── Stage 2: Cross-encoder ────────────────────────────────────────────────────

def _load_cross_encoder(model_name: str):
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            _cross_encoder = CrossEncoder(model_name, max_length=512)
            logger.info("Cross-encoder loaded: %s", model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — cross-encoder reranking disabled. "
                "pip install sentence-transformers"
            )
    return _cross_encoder


def rerank_facts(
    query: str,
    facts: list[dict],
    top_n: int | None = None,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[dict]:
    """
    Score each (query, fact_text) pair with the cross-encoder and return
    facts sorted by relevance score, best-first.

    Falls back to original order when sentence-transformers is not installed
    so the pipeline keeps working in minimal-dependency environments.

    Parameters
    ----------
    query      : the user's question
    facts      : list of dicts; each must have a 'fact_text' key
    top_n      : cap on returned facts (applied after sorting)
    model_name : cross-encoder model name (override for testing)

    Returns
    -------
    Re-ordered (and optionally trimmed) copy of facts.
    """
    if not facts:
        return facts

    encoder = _load_cross_encoder(model_name)
    if encoder is None:
        # Graceful degradation: just trim without reranking
        return facts[:top_n] if top_n else facts

    pairs = [(query, f["fact_text"]) for f in facts]
    scores = encoder.predict(pairs)  # returns numpy array, one score per pair

    ranked = sorted(zip(facts, scores), key=lambda x: float(x[1]), reverse=True)
    result = [f for f, _ in ranked]

    if top_n:
        result = result[:top_n]

    logger.info(
        "Cross-encoder reranked %d facts → returning top %d",
        len(facts),
        len(result),
    )
    return result
