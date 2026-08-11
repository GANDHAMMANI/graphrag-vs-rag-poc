"""
graphrag_store.py — GraphRAGStore, adapted from LlamaIndex's GraphRAG v2 cookbook.

Adds community detection (hierarchical Leiden, via graspologic) on top of the
Neo4j property graph, and generates an LLM summary per community. The original
cookbook hardcodes OpenAI() for summarization — swapped for our configured LLM
(Groq) so it doesn't need a separate API key.
"""

import re
from collections import defaultdict

import networkx as nx
from graspologic.partition import hierarchical_leiden
from llama_index.core.llms import LLM, ChatMessage
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore


class GraphRAGStore(Neo4jPropertyGraphStore):
    community_summary: dict = {}
    entity_info: dict | None = None
    max_cluster_size: int = 5
    summarizer_llm: LLM | None = None  # set after construction: store.summarizer_llm = llm

    def generate_community_summary(self, text: str) -> str:
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, "
                    "each represented as entity1->entity2->relation->relationship_description. "
                    "Your task is to create a summary of these relationships. The summary should "
                    "include the names of the entities involved and a concise synthesis of the "
                    "relationship descriptions. The goal is to capture the most critical and "
                    "relevant details that highlight the nature and significance of each "
                    "relationship. Ensure that the summary is coherent and integrates the "
                    "information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = self.summarizer_llm.chat(messages)
        return re.sub(r"^assistant:\s*", "", str(response)).strip()

    def build_communities(self):
        nx_graph = self._create_nx_graph()
        if nx_graph.number_of_nodes() == 0:
            print("GraphRAGStore: graph is empty, nothing to cluster.")
            return
        community_hierarchical_clusters = hierarchical_leiden(nx_graph, max_cluster_size=self.max_cluster_size)
        self.entity_info, community_info = self._collect_community_info(nx_graph, community_hierarchical_clusters)
        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        nx_graph = nx.Graph()
        triplets = self.get_triplets()
        for entity1, relation, entity2 in triplets:
            nx_graph.add_node(entity1.name)
            nx_graph.add_node(entity2.name)
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties.get("relationship_description", ""),
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        for item in clusters:
            node = item.node
            cluster_id = item.cluster
            entity_info[node].add(cluster_id)
            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                    community_info[cluster_id].append(detail)

        entity_info = {k: list(v) for k, v in entity_info.items()}
        return dict(entity_info), dict(community_info)

    def _summarize_communities(self, community_info):
        for community_id, details in community_info.items():
            details_text = "\n".join(details) + "."
            self.community_summary[community_id] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        if not self.community_summary:
            self.build_communities()
        return self.community_summary
