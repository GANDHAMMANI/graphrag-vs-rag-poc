"""
ingest.py — Build a proper GraphRAG index (LlamaIndex GraphRAG v2 pattern):
entity/relationship extraction WITH descriptions, Leiden community detection,
and per-community LLM summaries. See core/graphrag_extractor.py,
core/graphrag_store.py for what changed vs. the earlier plain PropertyGraphIndex
attempt (SimpleLLMPathExtractor, no clustering) — that version scored much
worse on RAGAs than our hand-built pipeline, this is the fair retest.

Usage:
    python ingest.py
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def main():
    from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    from core.config import settings
    from core.graphrag_extractor import KG_TRIPLET_EXTRACT_TMPL, GraphRAGExtractor, parse_fn
    from core.graphrag_store import GraphRAGStore
    from core.key_rotation import KeyRotator, RotatingGroq, load_keys

    keys = load_keys()
    logger.info("Loaded %d Groq API keys for rotation", len(keys))
    rotator = KeyRotator(keys)
    llm = RotatingGroq(rotator, model=settings.extraction_model, temperature=0)
    embed_model = HuggingFaceEmbedding(model_name=settings.embed_model)
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = settings.chunk_size
    Settings.chunk_overlap = settings.chunk_overlap

    graph_store = GraphRAGStore(
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        url=settings.neo4j_uri,
        database=settings.neo4j_database,
        connection_timeout=120,
        max_transaction_retry_time=60,
    )
    graph_store.summarizer_llm = llm

    # Clear previous run's graph — the earlier plain-PropertyGraphIndex attempt
    # wrote bare triples with no entity/relationship descriptions; mixing that
    # schema with this one would corrupt Leiden clustering and summaries.
    logger.info("Clearing existing graph …")
    graph_store.structured_query("MATCH (n) DETACH DELETE n")

    logger.info("Loading documents from %s …", DATA_DIR)
    documents = SimpleDirectoryReader(
        input_dir=str(DATA_DIR), recursive=True, required_exts=[".pdf", ".csv"]
    ).load_data()
    logger.info("Loaded %d documents", len(documents))

    # num_workers=2, max_paths_per_chunk=5 — conservative, Groq's per-minute
    # token limit gets blown through fast at higher extraction concurrency.
    kg_extractor = GraphRAGExtractor(
        llm=llm,
        extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        max_paths_per_chunk=5,
        num_workers=2,
        parse_fn=parse_fn,
    )

    logger.info("Extracting entities/relationships with descriptions (this calls the LLM per chunk) …")
    index = PropertyGraphIndex.from_documents(
        documents,
        llm=llm,
        embed_model=embed_model,
        kg_extractors=[kg_extractor],
        property_graph_store=graph_store,
        show_progress=True,
    )

    logger.info("Building communities (Leiden clustering + per-community LLM summaries) …")
    index.property_graph_store.build_communities()
    n_communities = len(index.property_graph_store.community_summary)
    logger.info("Built %d communities.", n_communities)

    logger.info("Ingestion complete. Graph is persisted directly in Neo4j at %s", settings.neo4j_uri)


if __name__ == "__main__":
    main()
