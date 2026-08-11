import logging

from neo4j import GraphDatabase

from .config import settings
from .embeddings import embed_text
from .rerank import rerank_facts, rrf_merge

logger = logging.getLogger(__name__)


class GraphRetriever:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        self._entity_vector_store = None  # lazy-init, only used when vector_backend="chroma"

    def close(self):
        self.driver.close()

    # ── Seed entity discovery ─────────────────────────────────────────────────

    def find_seed_entities_by_words(self, tx, query_text: str, limit: int = 3):
        stopwords = {
            "who", "is", "the", "a", "an", "what", "where", "when", "how",
            "does", "did", "do", "and", "or", "of", "in", "on", "for",
            "to", "was", "were", "are", "this", "that", "my", "his", "her",
        }
        words = [w for w in query_text.lower().split() if w not in stopwords and len(w) > 2]
        if not words:
            words = [query_text.lower()]

        # Order by how many query words each entity name matches — more matches = better rank
        result = tx.run(
            """
            MATCH (n:Entity)
            WHERE any(word IN $words WHERE toLower(n.name) CONTAINS word)
            WITH n, size([word IN $words WHERE toLower(n.name) CONTAINS word]) AS match_count
            ORDER BY match_count DESC
            RETURN DISTINCT n.name AS name
            LIMIT $limit
            """,
            words=words,
            limit=limit,
        )
        return [record["name"] for record in result]

    def find_seed_entities_by_meaning(self, tx, query_text: str, limit: int = 3, min_score: float = 0.45):
        query_vector = embed_text(query_text)
        result = tx.run(
            """
            CALL db.index.vector.queryNodes('entity_embeddings', $limit, $query_vector)
            YIELD node, score
            WHERE score >= $min_score
            RETURN node.name AS name, score
            """,
            query_vector=query_vector,
            limit=limit,
            min_score=min_score,
        )
        return [record["name"] for record in result]

    def find_seed_entities_by_meaning_external(self, query_text: str, limit: int = 3, min_score: float = 0.45):
        """Same contract as find_seed_entities_by_meaning, but backed by a separate
        Chroma vector store instead of Neo4j's native vector index — for comparing
        seed-lookup latency: embeddings on the graph node vs. in an external store."""
        from .entity_vector_store import EntityVectorStore
        if self._entity_vector_store is None:
            self._entity_vector_store = EntityVectorStore()
        return self._entity_vector_store.query(query_text, limit, min_score)

    def find_seed_entities(self, tx, query_text: str, limit: int = 3, vector_backend: str = "neo4j"):
        # Always run both word match and semantic search, then fuse with RRF —
        # word match alone can pick a plausible-looking but wrong seed and
        # never get corrected. Costs ~300-500ms more per query than the old
        # word-match-only-when-confident shortcut, traded for better recall.
        #
        # vector_backend picks where the semantic half looks up entity
        # embeddings: "neo4j" (native vector index, default/production path)
        # or "chroma" (separate vector store, for latency comparison).
        word_matches = self.find_seed_entities_by_words(tx, query_text, limit)
        if vector_backend == "chroma":
            semantic_matches = self.find_seed_entities_by_meaning_external(query_text, limit)
        else:
            semantic_matches = self.find_seed_entities_by_meaning(tx, query_text, limit)
        merged = rrf_merge([word_matches, semantic_matches])
        logger.debug("Seed entities (RRF, backend=%s): word=%s semantic=%s fused=%s", vector_backend, word_matches, semantic_matches, merged[:limit])
        return merged[:limit]

    # ── Context expansion ─────────────────────────────────────────────────────

    def expand_context(self, tx, entity_name: str, hops: int = 2) -> list[str]:
        """Return plain text facts (original format, kept for app.py)."""
        result = tx.run(
            f"""
            MATCH path = (start:Entity {{name: $name}})-[*1..{hops}]-(connected)
            UNWIND relationships(path) AS rel
            RETURN DISTINCT
                startNode(rel).name AS subject,
                type(rel)          AS relationship,
                endNode(rel).name  AS object,
                rel.page_number    AS page_number,
                rel.source         AS source
            """,
            name=entity_name,
        )
        return [
            f"{r['subject']} {r['relationship'].replace('_', ' ')} {r['object']}"
            for r in result
        ]

    def expand_context_structured(self, tx, entity_name: str, hops: int = 2) -> list[dict]:
        """
        Return structured fact dicts including provenance from relationship properties.

        Each dict: {subject, relationship, object, page_number, source, bbox, fact_text}
        bbox is a list [x0, y0, x1, y1] in PDF-point coordinates, or None.
        """
        result = tx.run(
            f"""
            MATCH path = (start:Entity {{name: $name}})-[*1..{hops}]-(connected)
            UNWIND relationships(path) AS rel
            RETURN DISTINCT
                startNode(rel).name AS subject,
                type(rel)          AS relationship,
                endNode(rel).name  AS object,
                rel.page_number    AS page_number,
                rel.source         AS source,
                rel.bbox           AS bbox,
                rel.section        AS section
            """,
            name=entity_name,
        )
        facts = []
        seen = set()
        for r in result:
            key = (r["subject"], r["relationship"], r["object"])
            if key in seen:
                continue
            seen.add(key)
            facts.append({
                "subject": r["subject"],
                "relationship": r["relationship"],
                "object": r["object"],
                "page_number": r["page_number"],
                "source": r["source"] or "",
                "bbox": list(r["bbox"]) if r["bbox"] is not None else None,
                "section": r["section"] or "",
                "fact_text": f"{r['subject']} {r['relationship'].replace('_', ' ')} {r['object']}",
            })
        return facts

    # ── Public retrieval API ──────────────────────────────────────────────────

    def retrieve_context(self, query_text: str, hops: int = 2) -> str:
        """Plain-text context string — kept for backward compat with app.py."""
        with self.driver.session() as session:
            seeds = session.execute_read(self.find_seed_entities, query_text)
            if not seeds:
                return ""
            all_facts: set[str] = set()
            for seed in seeds:
                facts = session.execute_read(self.expand_context, seed, hops)
                all_facts.update(facts)
            return "\n".join(f"- {fact}" for fact in sorted(all_facts))

    def retrieve_with_citations(self, query_text: str, hops: int = 2, max_facts: int | None = None) -> dict:
        """
        Structured retrieval result for the FastAPI layer.

        Returns:
            {
              "context":   str  — bulleted facts for LLM prompt,
              "citations": list[dict] — per-fact provenance (page, source),
              "sources":   list[str] — deduplicated source file names,
            }
        """
        with self.driver.session() as session:
            seeds = session.execute_read(self.find_seed_entities, query_text)
            if not seeds:
                logger.info("No seed entities found for query: %s", query_text)
                return {"context": "", "citations": [], "sources": []}

            all_facts: list[dict] = []
            seen_keys: set[tuple] = set()

            for seed in seeds:
                facts = session.execute_read(self.expand_context_structured, seed, hops)
                for fact in facts:
                    key = (fact["subject"], fact["relationship"], fact["object"])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_facts.append(fact)

        logger.info("Raw facts before dedup: %d", len(all_facts))

        # Deduplicate by (subject, relationship, object)
        seen_triples: set[tuple] = set()
        unique_facts: list[dict] = []
        for f in all_facts:
            key = (f["subject"], f["relationship"], f["object"])
            if key not in seen_triples:
                seen_triples.add(key)
                unique_facts.append(f)
        all_facts = unique_facts
        logger.info("After triple dedup: %d", len(all_facts))

        # Keyword pre-rank — fast, no model inference
        query_words = set(query_text.lower().split())
        all_facts.sort(
            key=lambda f: sum(1 for w in query_words if w in f["fact_text"].lower()),
            reverse=True,
        )

        # Cross-encoder only when fact pool is large enough to benefit from it.
        # For small sets keyword ranking is already good — skipping saves ~400ms.
        if len(all_facts) > 20:
            all_facts = rerank_facts(
                query_text,
                all_facts[:40],
                top_n=max_facts,
                model_name=settings.rerank_model,
            )
        else:
            all_facts = all_facts[:max_facts] if max_facts else all_facts
            logger.info("Skipped cross-encoder (only %d facts)", len(all_facts))

        context_lines = [f["fact_text"] for f in all_facts]
        context = "\n".join(f"- {line}" for line in context_lines)

        # Build citations — deduplicate by (page_number, section, bbox).
        # When bbox is null, we still differentiate by section so facts from
        # "Projects" and "Experience" on the same page produce separate citation cards.
        seen_locations: set[tuple] = set()
        citations: list[dict] = []
        for f in all_facts:
            if not (f.get("page_number") or f.get("source")):
                continue
            bbox = f.get("bbox")
            section = f.get("section") or ""
            loc_key = (f.get("page_number"), section, tuple(bbox) if bbox else "null")
            if loc_key in seen_locations:
                continue
            seen_locations.add(loc_key)
            citations.append({
                "fact": f["fact_text"],
                "page_number": f["page_number"],
                "source": f["source"],
                "bbox": bbox,
                "section": section,
            })

        sources = sorted({f["source"] for f in all_facts if f.get("source")})

        logger.info(
            "retrieve_with_citations: %d unique facts, %d unique citations, %d sources",
            len(all_facts), len(citations), len(sources),
        )
        return {"context": context, "citations": citations, "sources": sources}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    retriever = GraphRetriever()

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter a question: ")
    print(f"Question: {question}\n")

    result = retriever.retrieve_with_citations(question)
    print("Context:\n", result["context"] or "(no matching entities found)")
    print(f"\nCitations ({len(result['citations'])}):")
    for c in result["citations"][:10]:
        print(f"  [{c['source']} p.{c['page_number']}] {c['fact']}")

    retriever.close()
