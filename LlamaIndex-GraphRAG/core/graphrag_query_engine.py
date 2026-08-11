"""
graphrag_query_engine.py — GraphRAGQueryEngine, adapted from LlamaIndex's GraphRAG v2
cookbook.

Retrieval strategy: find entities relevant to the question, map them to their
community (from GraphRAGStore's Leiden clustering), rank those communities by
embedding similarity to the query, generate a partial answer per top-ranked
community, then synthesize those into one final answer.
"""

import logging
import re

import numpy as np
from llama_index.core import PropertyGraphIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.query_engine import CustomQueryEngine

from .graphrag_store import GraphRAGStore

logger = logging.getLogger(__name__)


class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    index: PropertyGraphIndex
    llm: LLM
    embed_model: BaseEmbedding
    similarity_top_k: int = 10
    top_k_communities: int = 3  # cap how many community summaries get a full LLM pass

    def custom_query(self, query_str: str) -> str:
        return self.query_with_context(query_str)["answer"]

    def query_with_context(self, query_str: str) -> dict:
        entities = self.get_entities(query_str, self.similarity_top_k)
        community_ids = self.retrieve_entity_communities(self.graph_store.entity_info, entities)
        community_summaries = self.graph_store.get_community_summaries()
        candidates = {cid: s for cid, s in community_summaries.items() if cid in community_ids}

        if not candidates:
            return {"answer": "No relevant information found in the knowledge graph.", "contexts": []}

        used_summaries = self._rank_communities(query_str, candidates)

        community_answers = []
        for s in used_summaries:
            try:
                community_answers.append(self.generate_answer_from_summary(s, query_str))
            except Exception as e:
                logger.warning("Skipping a community summary after LLM call failure: %s", e)

        # Drop per-community answers that are themselves hedges/non-answers —
        # letting these into aggregate_answers is what caused the model to
        # blend a correct answer with "not found" and hedge on the final output.
        concrete_answers = [a for a in community_answers if not self._is_non_answer(a)]

        if not concrete_answers:
            # every community either failed or found nothing — genuinely no answer
            return {
                "answer": community_answers[0] if community_answers else "No relevant information found in the knowledge graph.",
                "contexts": used_summaries,
            }

        if len(concrete_answers) == 1:
            return {"answer": concrete_answers[0], "contexts": used_summaries}

        return {"answer": self.aggregate_answers(concrete_answers), "contexts": used_summaries}

    @staticmethod
    def _is_non_answer(answer: str) -> bool:
        """Heuristic check for 'no information found'-style hedges, so they
        can be excluded before aggregation instead of diluting a correct
        answer found in a different community."""
        markers = (
            "no information", "not mention", "does not provide",
            "not available", "cannot determine", "no relevant information",
            "not possible to determine", "not specify",
        )
        lowered = answer.lower()
        return any(m in lowered for m in markers)

    def get_entities(self, query_str: str, similarity_top_k: int) -> list[str]:
        nodes_retrieved = self.index.as_retriever(similarity_top_k=similarity_top_k).retrieve(query_str)
        entities = set()
        pattern = r"^([\w\-'&.]+(?:\s+[\w\-'&.]+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*([\w\-'&.]+(?:\s+[\w\-'&.]+)*)$"
        for node in nodes_retrieved:
            for match in re.findall(pattern, node.text, re.MULTILINE | re.IGNORECASE):
                entities.add(match[0])
                entities.add(match[2])
        return list(entities)

    def retrieve_entity_communities(self, entity_info: dict | None, entities: list[str]) -> list:
        if not entity_info:
            return []
        community_ids = []
        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])
        return list(set(community_ids))

    def _rank_communities(self, query_str: str, candidates: dict) -> list[str]:
        """Embed the query and each candidate summary, keep the top_k_communities
        by cosine similarity — avoids passing every loosely-connected community
        into the answer, which was diluting answer_relevancy."""
        if len(candidates) <= self.top_k_communities:
            return list(candidates.values())

        query_emb = np.array(self.embed_model.get_query_embedding(query_str))
        scored = []
        for summary in candidates.values():
            summary_emb = np.array(self.embed_model.get_text_embedding(summary))
            sim = float(
                np.dot(query_emb, summary_emb)
                / (np.linalg.norm(query_emb) * np.linalg.norm(summary_emb) + 1e-8)
            )
            scored.append((sim, summary))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[: self.top_k_communities]]

    def generate_answer_from_summary(self, community_summary: str, query: str) -> str:
        prompt = f"Given the community summary: {community_summary}, how would you answer the following query? Query: {query}"
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content="I need an answer based on the above information."),
        ]
        response = self.llm.chat(messages)
        return re.sub(r"^assistant:\s*", "", str(response)).strip()

    def aggregate_answers(self, community_answers: list[str]) -> str:
        prompt = (
            "Combine the following intermediate answers into a single, concise final answer. "
            "All of these answers contain real information — do not hedge or claim information "
            "is missing. If the answers cover different aspects of the question, merge them. "
            "If they overlap, keep the most complete and specific version."
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=f"Intermediate answers: {community_answers}"),
        ]
        try:
            response = self.llm.chat(messages)
        except Exception as e:
            logger.warning("aggregate_answers failed, falling back to first partial answer: %s", e)
            return community_answers[0]
        return re.sub(r"^assistant:\s*", "", str(response)).strip()