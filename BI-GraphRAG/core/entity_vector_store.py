"""
entity_vector_store.py — Entity embeddings in a separate vector store (Chroma),
as an alternative to Neo4j's native vector index (n.embedding + entity_embeddings
index, see core/load.py).

Same embeddings, same entity names — just stored and queried outside the graph,
so we can compare seed-entity lookup latency: Neo4j-native vs external store.

Usage:
    python -m core.entity_vector_store   # (re)builds the collection from Neo4j
"""

import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from neo4j import GraphDatabase

from .config import settings

logger = logging.getLogger(__name__)

_CHROMA_PATH = str(Path(__file__).parent.parent / "chroma_db")
_COLLECTION = "entity_embeddings"


def _get_collection():
    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=settings.embed_model)
    return client.get_or_create_collection(
        name=_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def build_entity_vector_store() -> int:
    """
    Pull every Entity node's (name, embedding) straight out of Neo4j — they're
    already computed at ingestion time — and load them into a Chroma collection.
    Avoids recomputing embeddings; just relocates them to a separate store.
    """
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password))
    with driver.session() as session:
        rows = session.run(
            "MATCH (n:Entity) WHERE n.embedding IS NOT NULL RETURN n.name AS name, n.embedding AS embedding"
        ).data()
    driver.close()

    if not rows:
        logger.warning("No embedded entities found in Neo4j — run ingestion first.")
        return 0

    # Dedupe by name — same entity name can exist as multiple graph nodes
    # (not merged into one), but Chroma requires unique IDs.
    seen = set()
    unique_rows = []
    for r in rows:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique_rows.append(r)
    rows = unique_rows

    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    try:
        client.delete_collection(_COLLECTION)
    except Exception:
        pass
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=settings.embed_model)
    collection = client.create_collection(name=_COLLECTION, embedding_function=ef, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[r["name"] for r in rows],
        embeddings=[r["embedding"] for r in rows],
        documents=[r["name"] for r in rows],
    )
    logger.info("Loaded %d entity embeddings into Chroma collection '%s'", len(rows), _COLLECTION)
    return len(rows)


class EntityVectorStore:
    """Query wrapper mirroring GraphRetriever.find_seed_entities_by_meaning's contract."""

    def __init__(self):
        self.collection = _get_collection()

    def query(self, query_text: str, limit: int = 3, min_score: float = 0.45) -> list[str]:
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_texts=[query_text],
            n_results=min(limit, count),
            include=["distances"],
        )
        names = result["ids"][0]
        distances = result["distances"][0]
        # Chroma cosine distance -> similarity score, same scale as Neo4j's cosine score
        return [name for name, dist in zip(names, distances) if (1 - dist) >= min_score]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    n = build_entity_vector_store()
    print(f"Indexed {n} entities into Chroma.")
