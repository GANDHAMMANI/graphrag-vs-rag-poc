"""
retrieve.py — Query the GraphRAG index built by ingest.py, using the community-
based retrieval strategy from LlamaIndex's GraphRAG v2 cookbook (not the default
PropertyGraphIndex query engine).

Community summaries are in-memory on the GraphRAGStore Python object, not
persisted to Neo4j — so they get rebuilt once per process, at __init__, and
reused across every query in that process's lifetime (matches how
eval/evaluate.py and app.py use this: one retriever instance for the whole run).

Uses two separate LLMs: extraction_model for community summarization (matches
what ingest.py uses, since summarization quality feeds every downstream
answer) and answer_model for query-time answer generation/aggregation.
"""

import logging

from .config import settings

logger = logging.getLogger(__name__)


class LlamaIndexGraphRetriever:
    def __init__(self):
        from llama_index.core import PropertyGraphIndex, Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        from .graphrag_query_engine import GraphRAGQueryEngine
        from .graphrag_store import GraphRAGStore
        from .key_rotation import KeyRotator, RotatingGroq, load_keys

        rotator = KeyRotator(load_keys())
        self.answer_llm = RotatingGroq(rotator, model=settings.extraction_model, temperature=0.1)
        self.summarizer_llm = RotatingGroq(rotator, model=settings.extraction_model, temperature=0)
        embed_model = HuggingFaceEmbedding(model_name=settings.embed_model)
        Settings.llm = self.answer_llm
        Settings.embed_model = embed_model

        graph_store = GraphRAGStore(
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            url=settings.neo4j_uri,
            database=settings.neo4j_database,
            connection_timeout=120,
        )
        graph_store.summarizer_llm = self.summarizer_llm

        self.index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            llm=self.answer_llm,
            embed_model=embed_model,
        )

        logger.info("Building communities for this session (rebuilt per-process, not persisted) …")
        graph_store.build_communities()
        logger.info("Built %d communities.", len(graph_store.community_summary))

        self.query_engine = GraphRAGQueryEngine(
            graph_store=graph_store,
            index=self.index,
            llm=self.answer_llm,
            embed_model=embed_model,
            similarity_top_k=10,
        )

    def retrieve_with_citations(self, query_text: str) -> dict:
        result = self.query_engine.query_with_context(query_text)
        citations = [{"text": c[:200], "source": "community_summary", "page_number": None} for c in result["contexts"]]
        return {
            "answer": result["answer"],
            "citations": citations,
            "contexts": result["contexts"],
        }

    def close(self):
        pass