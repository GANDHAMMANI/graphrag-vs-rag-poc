"""
graphrag_extractor.py — GraphRAGExtractor, adapted from LlamaIndex's GraphRAG v2
cookbook: https://developers.llamaindex.ai/python/examples/cookbooks/graphrag_v2/

Unlike SimpleLLMPathExtractor (bare subject-relation-object triples, no context),
this extracts entity TYPES and DESCRIPTIONS, and relationship DESCRIPTIONS —
the richer signal the community-summarization step downstream depends on.
"""

import asyncio
import json
import logging
import re
from typing import Any, Callable, List, Optional, Union

from llama_index.core import Settings
from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY, EntityNode, Relation
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.schema import BaseNode, TransformComponent

logger = logging.getLogger(__name__)

KG_TRIPLET_EXTRACT_TMPL = """-Goal-
Given a text document, identify all entities and their entity types from the text
and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, identify all pairs of (source_entity,
target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and
the target entity are related to each other

3. Output Formatting:
- Return the result in valid JSON format with two keys: 'entities' (list of entity
objects) and 'relationships' (list of relationship objects).
- Exclude any text outside the JSON structure (e.g., no explanations or comments).
- If no entities or relationships are identified, return empty lists:
{{ "entities": [], "relationships": [] }}.

-An Output Example-
{{
  "entities": [
    {{
      "entity_name": "Albert Einstein",
      "entity_type": "Person",
      "entity_description": "Albert Einstein was a theoretical physicist who developed
the theory of relativity and made significant contributions to physics."
    }}
  ],
  "relationships": [
    {{
      "source_entity": "Albert Einstein",
      "target_entity": "Theory of Relativity",
      "relation": "developed",
      "relationship_description": "Albert Einstein is the developer of the theory of relativity."
    }}
  ]
}}

-Real Data-
######################
text: {text}
######################
output:"""


def parse_fn(response_str: str) -> Any:
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, response_str, re.DOTALL)
    entities, relationships = [], []
    if not match:
        logger.warning("GraphRAGExtractor: no JSON object found in LLM output (chunk contributed 0 entities)")
        return entities, relationships
    try:
        data = json.loads(match.group(0))
        entities = [
            (e["entity_name"], e["entity_type"], e["entity_description"])
            for e in data.get("entities", [])
        ]
        relationships = [
            (r["source_entity"], r["target_entity"], r["relation"], r["relationship_description"])
            for r in data.get("relationships", [])
        ]
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(
            "GraphRAGExtractor: failed to parse LLM output (%s) — chunk contributed 0 entities. Raw: %.200s",
            e, response_str,
        )
    return entities, relationships


class GraphRAGExtractor(TransformComponent):
    """Extract entity/relationship triples with descriptions, via LLM + JSON parsing."""

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = parse_fn,
        max_paths_per_chunk: int = 5,
        num_workers: int = 2,
    ) -> None:
        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)
        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphRAGExtractor"

    def __call__(self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any) -> List[BaseNode]:
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def _aextract(self, node: BaseNode) -> BaseNode:
        assert hasattr(node, "text")
        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt, text=text, max_knowledge_triplets=self.max_paths_per_chunk
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except Exception as e:
            logger.warning("GraphRAGExtractor: extraction failed on a chunk: %s", e)
            entities, entities_relationship = [], []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        # NOTE: metadata dict is copied INSIDE each loop iteration — a shared
        # dict here means every entity/relation in the chunk ends up pointing
        # at the same object, so only the last write survives (fixed bug).
        for entity, entity_type, description in entities:
            entity_metadata = node.metadata.copy()
            entity_metadata["entity_description"] = description
            existing_nodes.append(EntityNode(name=entity, label=entity_type, properties=entity_metadata))

        for subj, obj, rel, description in entities_relationship:
            relation_metadata = node.metadata.copy()
            relation_metadata["relationship_description"] = description
            existing_relations.append(
                Relation(label=rel, source_id=subj, target_id=obj, properties=relation_metadata)
            )

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any) -> List[BaseNode]:
        jobs = [self._aextract(node) for node in nodes]
        return await run_jobs(jobs, workers=self.num_workers, show_progress=show_progress, desc="Extracting paths from text")