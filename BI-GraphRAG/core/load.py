import logging

from neo4j import GraphDatabase

from .config import settings
from .embeddings import embed_text

logger = logging.getLogger(__name__)


class GraphLoader:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )

    def close(self):
        self.driver.close()

    def load_triple(self, tx, triple: dict):
        subject_type = _sanitize_label(triple.get("subject_type", "Entity"))
        object_type = _sanitize_label(triple.get("object_type", "Entity"))
        relationship = _sanitize_relationship(triple["relationship"])

        # MERGE keys are unchanged — dedup behaviour is preserved.
        # ON CREATE SET adds provenance only when the relationship is first seen;
        # re-running the same document won't overwrite existing metadata.
        query = f"""
        MERGE (s:Entity:{subject_type} {{name: $subject}})
        MERGE (o:Entity:{object_type} {{name: $object}})
        MERGE (s)-[r:{relationship}]->(o)
        ON CREATE SET r.page_number = $page_number, r.source = $source, r.bbox = $bbox, r.section = $section
        """
        tx.run(
            query,
            subject=triple["subject"],
            object=triple["object"],
            page_number=triple.get("page_number"),
            source=triple.get("source", ""),
            bbox=triple.get("bbox"),
            section=triple.get("section"),
        )

        tx.run(
            "MATCH (n:Entity {name: $name}) SET n.embedding = $embedding",
            name=triple["subject"],
            embedding=embed_text(triple["subject"]),
        )
        tx.run(
            "MATCH (n:Entity {name: $name}) SET n.embedding = $embedding",
            name=triple["object"],
            embedding=embed_text(triple["object"]),
        )

    def ensure_vector_index(self):
        """
        Create the vector index once (IF NOT EXISTS makes it a no-op after the
        first run). Lets Neo4j do similarity search natively.
        """
        with self.driver.session() as session:
            session.run(
                """
                CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                FOR (n:Entity) ON (n.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
                """
            )

    def _write_all_triples(self, tx, triples: list[dict]):
        """Write every triple in a single transaction — one round-trip instead of N."""
        for triple in triples:
            self.load_triple(tx, triple)

    def _batch_embed(self, tx, names: list[str]):
        """
        Compute embeddings in Python, then push all of them to Neo4j in one
        UNWIND query instead of 2 queries per triple.
        """
        rows = [{"name": n, "embedding": embed_text(n)} for n in names]
        tx.run(
            "UNWIND $rows AS row "
            "MATCH (n:Entity {name: row.name}) "
            "SET n.embedding = row.embedding",
            rows=rows,
        )

    def load_triples(self, triples: list[dict]):
        if not triples:
            return
        self.ensure_vector_index()
        with self.driver.session() as session:
            # All MERGE/ON CREATE in one transaction (was N separate transactions)
            session.execute_write(self._write_all_triples, triples)
            # All embeddings in one UNWIND query (was 2 queries × N triples)
            unique_names = list({t["subject"] for t in triples} | {t["object"] for t in triples})
            session.execute_write(self._batch_embed, unique_names)
        logger.info("Loaded %d triples, %d entities embedded.", len(triples), len(unique_names))

    def clear_all(self):
        """Wipe the graph — handy while iterating on a POC."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared.")


def _sanitize_label(label: str) -> str:
    return "".join(c for c in label if c.isalnum() or c == "_") or "Entity"


def _sanitize_relationship(rel: str) -> str:
    cleaned = "".join(c for c in rel if c.isalnum() or c == "_")
    return cleaned.upper() or "RELATED_TO"


if __name__ == "__main__":
    import sys
    from core.extract import extract_triples

    logging.basicConfig(level=logging.INFO)

    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter text to extract and load: ")

    logger.info("Extracting triples...")
    triples = extract_triples(text)
    logger.info("%s", triples)

    logger.info("Loading into Neo4j...")
    loader = GraphLoader()
    loader.load_triples(triples)
    loader.close()
