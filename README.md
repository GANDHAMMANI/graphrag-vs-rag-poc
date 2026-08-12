# GraphRAG vs Traditional RAG: A Comparative Evaluation

A comparative study of three retrieval-augmented generation approaches  a custom-built GraphRAG pipeline, a traditional chunk-based RAG pipeline, and a framework-based GraphRAG implementation (LlamaIndex) evaluated on the same dataset and question set.

> **Scope note**: This evaluation was conducted on a small, synthetic business dataset (30 ground-truth questions). Results should be treated as directional findings from a proof-of-concept, not as a production benchmark.

---

## 1. Overview

Two projects are included in this repository:

- **[`BI-GraphRAG/`](./BI-GraphRAG)** : A custom-built GraphRAG pipeline (Neo4j-based) compared directly against a Traditional RAG pipeline (ChromaDB-based) on the same data, via a side-by-side API.
- **[`LlamaIndex-GraphRAG/`](./LlamaIndex-GraphRAG)** : The same comparison rebuilt using LlamaIndex's official GraphRAG pattern (rich entity/relationship extraction, Leiden community clustering, community-summary retrieval).

Both projects use the same underlying dataset and the same 30-question ground truth, scored with [RAGAs](https://github.com/explodinggradients/ragas) using an LLM judge (`llama-3.3-70b-versatile`).
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/4bdc2068-ff50-4baf-bbe8-504571d793d4" />




**How the two GraphRAG approaches actually differ:** the difference isn't just extraction quality : it's *when* and *how* each system reads the graph at answer time.

- **Custom GraphRAG retrieves live.** Given a question, it finds a starting entity and traverses outward through the graph, hop by hop, collecting every connected fact along the way. The LLM then answers directly from that raw set of facts.
- **LlamaIndex GraphRAG retrieves from pre-computed summaries.** At ingestion time, it clusters the graph into tightly-connected neighborhoods ("communities") using Leiden clustering, and has an LLM write a summary for each one *before* any question is asked. At query time, it identifies which communities are relevant, pulls their summaries, generates a partial answer per community, and synthesizes those into a final answer  it never traverses the graph live.

In short: one explores the graph on demand; the other consults notes written about the graph in advance.



## 2. Methodology

### Dataset

| Source | Contents |
|---|---|
| Structured (CSV) | Company, departments, regions, products, employees, customers, orders |
| Unstructured (PDF) | Meeting notes, quarterly business reviews (QBRs) |
| Ground truth | 30 hand-written questions spanning relational lookups and narrative facts |

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/81d7b390-5d34-4395-9823-9a0a86f32470" />

### Systems compared

| | Traditional RAG | Custom GraphRAG | LlamaIndex GraphRAG |
|---|---|---|---|
| Ingestion | Chunk → embed → ChromaDB | Extract triples (LLM + regex) → Neo4j | Extract entities/relationships with descriptions → Neo4j |
| Structuring | None | None beyond triples | Leiden clustering into communities, LLM-summarized |
| Retrieval | Cosine similarity over chunks | Seed entity search (keyword + semantic, RRF-fused) → 2-hop graph traversal → cross-encoder rerank | Seed entity search → map to communities → retrieve relevant community summaries |
| Answer generation | LLM answers from retrieved chunks | LLM answers from retrieved facts | LLM synthesizes partial answers per community into a final answer |

### PDF ingestion

PDF text and layout extraction was handled by **OpenDataLoader PDF Hybrid** (`docling-fast` backend), used consistently across both projects rather than a generic PDF-to-text extractor. This was chosen for its structure awareness  it returns typed elements (headings, tables, body text) with page numbers and bounding boxes attached, rather than a single undifferentiated text blob per page. That structure was load-bearing for both the chunking strategy below and for citation/provenance tracking (page number + bounding box per retrieved fact).

### Chunking strategy

Both projects used a **structure-aware chunking** approach rather than fixed-size splitting alone:

- Headings, when present, open new sections; body text accumulates under the current section until the next heading.
- Tables are always extracted as their own standalone chunk, never merged with surrounding text.
- For documents without clear headings (e.g. flat-format notes), section boundaries were inferred from significant vertical gaps between text blocks on the page.
- Long sections were split with a sliding window (chunk size 1200 characters, 150-character overlap), with page number and bounding box tracked per character offset  so a sub-chunk carries the location of the specific text it starts in, not just the location of the first element in its section.

### Retrieval techniques

**Traditional RAG** used a two-stage retrieval: cosine similarity search over chunk embeddings to get an initial candidate pool, followed by a cross-encoder reranking pass, with a minimum relevance score threshold applied before any chunk was passed to the LLM.

**Custom GraphRAG** used a three-stage retrieval:
1. **Seed entity discovery** : run in parallel via keyword matching (substring match against entity names, stopwords removed) and semantic matching (embedding similarity against entity embeddings stored in Neo4j's native vector index), then fused with **Reciprocal Rank Fusion (RRF)** so entities found by either method contribute to the final ranking.
2. **Graph traversal** : from each seed entity, walk outward up to 2 hops, collecting every connected fact along the way.
3. **Reranking** : a keyword pre-rank, followed by the same cross-encoder used in the Traditional RAG lane when the candidate pool was large enough to benefit from it.

**LlamaIndex GraphRAG** used similarity-based entity retrieval (via LlamaIndex's built-in retriever) to identify relevant entities, mapped those entities to their pre-computed communities, and retrieved the community summaries for synthesis  described in detail below.

### Fairness measures across lanes

Two consistency measures were applied so the comparison isolated retrieval strategy as the variable under test, rather than incidental implementation differences:

- **Identical ID-to-name resolution for CSV data.** Both the Traditional RAG and Custom GraphRAG ingestion paths resolve foreign-key IDs (e.g. `region_id`, `department_id`) to their human-readable names *before* embedding or storing. Without this, a question like "who works in North America" would be unanswerable by whichever lane stored the raw ID instead of the name  the resolution logic was deliberately duplicated across both ingestion scripts to keep this parity.
- **Matched answer-generation model.** All lanes were confirmed to use the same model for final answer generation before scoring, after an earlier run surfaced a mismatch (see Section 4) that would otherwise have confounded the comparison.

### Provenance and citations

Every retrieved fact (GraphRAG) or chunk (Traditional RAG) carries its source document, page number, and where available bounding box coordinates through to the API response, so answers can be traced back to the exact location in the source document they came from. This was implemented consistently across both lanes rather than added to one and not the other.

### Evaluation infrastructure: judge API key rotation

RAGAs scoring is judge-LLM-call-intensive: 30 questions × 4 metrics × 3 systems, plus retries on transient failures, adds up to several hundred Groq API calls per full evaluation pass  each carrying a non-trivial amount of context (the retrieved facts/chunks plus the ground truth). Groq's free tier enforces both a per-minute request cap (30 requests/minute) and a per-key daily token budget, either of which a single API key can exhaust partway through a full scoring run.

To avoid this, judge calls were rotated across a pool of API keys rather than relying on one. This served two purposes: it pooled the available request/token budget across the whole run instead of a single key's limit, and it made it practical to use the strongest available judge model (`llama-3.3-70b-versatile`) for scoring consistency, rather than defaulting to a smaller model purely to conserve one key's quota. Key rotation reduced  but did not eliminate  rate-limit failures; the retrieval-strategy benchmark scoring run (Section 5) still hit exhausted quota mid-run, which is why those results are not included above.

### Evaluation metrics (RAGAs)

| Metric | What it measures |
|---|---|
| Faithfulness | Whether the answer is fully supported by the retrieved context |
| Answer Relevancy | Whether the answer addresses the question asked |
| Context Precision | Proportion of retrieved context that is actually relevant |
| Context Recall | Proportion of required ground-truth facts that were successfully retrieved |


## 3. Results


### Final comparison

| Metric | Custom GraphRAG | Traditional RAG | LlamaIndex GraphRAG |
|---|---|---|---|
| Faithfulness | 0.872 | 0.794 | 0.822 |
| Answer Relevancy | 0.341 | 0.532 | 0.345 |
| Context Precision | 0.190 | 0.672 | 0.400 |
| Context Recall | 0.389 | 0.578 | 0.389 |

### Key observations

- **Faithfulness** was comparable across all three systems (0.79–0.87), with no clear structural advantage for any single approach.
- **Answer Relevancy** favored Traditional RAG (0.532) over both graph-based approaches, which scored nearly identically to each other (0.341 / 0.345).
- **Context Precision** was the largest and most consistent gap: both graph-based systems retrieved a higher proportion of low-relevance context than Traditional RAG, with LlamaIndex GraphRAG (0.400) narrowing  but not closing the gap seen in Custom GraphRAG (0.190).
- **Context Recall** was also comparable between the two graph-based systems (0.389 each), both trailing Traditional RAG (0.578).

**Overall finding**: across both graph-based implementations, retrieving discrete facts/triples consistently underperformed retrieving full text chunks on Context Precision, independent of extraction richness or retrieval strategy (hop-traversal vs. community summarization). This suggests the gap is at least partly structural to fact-based context rather than attributable to implementation choices alone.

### Framework alone did not guarantee GraphRAG quality

Before arriving at the LlamaIndex numbers above, an initial implementation using LlamaIndex's default extraction (`SimpleLLMPathExtractor`, bare triples, no clustering) was evaluated for comparison:

| Metric | LlamaIndex (naive, default config) | LlamaIndex (proper  extraction + clustering + fixes) |
|---|---|---|
| Faithfulness | 0.249 | 0.822 |
| Answer Relevancy | 0.503 | 0.345 |
| Context Precision | 0.256 | 0.400 |
| Context Recall | 0.206 | 0.389 |

This progression is why the final numbers above should not be read as "LlamaIndex GraphRAG" in the abstract  the framework provides the building blocks (richer extraction, clustering, community-summary retrieval), but implementation choices within those building blocks materially determined the outcome. See Section 4 for the specific issues that separated these two runs.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/fd69f05d-fc21-4490-b428-0fea85d73fe6" />

*Illustrative example  not drawn from the actual dataset. Shown to explain the mechanism, not to represent an actual retrieval result.*



## 4. Issues identified and resolved

Several implementation issues were identified during evaluation and corrected prior to final scoring. Each materially affected results.

| Issue | Fix | Impact |
|---|---|---|
| Entity nodes were keyed on name **and** type in Neo4j, causing the same real-world entity to fragment into disconnected nodes when tagged inconsistently across documents | Merge on name only; store type as a property | Reduced silent recall loss in graph traversal |
| Entity/relationship descriptions in the LlamaIndex extractor were written to a shared, reused dictionary, so all entities from a chunk inherited the last entity's description | Copy the metadata object per entity/relationship | Faithfulness 0.685 → 0.822, Context Precision 0.265 → 0.400 |
| LlamaIndex query engine generated and blended partial answers from every community touched by retrieved entities, with no relevance filtering | Filter to the most relevant communities before synthesis | Answer Relevancy 0.245 → 0.345 |
| Traditional RAG and GraphRAG lanes used different answer-generation models (70B vs. 8B), confounding the comparison | Matched models across both lanes | Removed a confounding variable from all subsequent runs |

### Engineering notes

Two additional implementation details worth noting, though they did not directly affect the evaluation metrics above:

- **Concurrent ingestion.** PDF extraction and LLM-based triple/entity extraction were parallelized across documents during ingestion, reducing total ingestion time substantially compared to sequential processing  relevant mainly for reproducing these results faster, not for retrieval quality itself.
- **Vector store placement experiment.** A side experiment compared storing entity embeddings natively on Neo4j graph nodes (via Neo4j's vector index) against storing them in a separate ChromaDB collection, to evaluate seed-entity lookup latency under each approach. This was a latency comparison only and did not factor into the quality metrics reported in Section 3.



## 5. Limitations

- Evaluation was run on a small, synthetic dataset; findings may not generalize to larger or real-world corpora.
- Each configuration was scored in a single evaluation pass; run-to-run judge variance was not measured.
- A secondary benchmark comparing seed-entity retrieval strategies (keyword-only, semantic-only, hybrid) was inconclusive due to LLM judge API rate limits during scoring and is not included in the results above.
- This is a proof-of-concept comparison, not a production readiness assessment.


## 6. Repository structure

```
.
├── README.md
├── BI-GraphRAG/              Custom GraphRAG + Traditional RAG (Neo4j + ChromaDB)
│   └── README.md             Setup and run instructions
├── LlamaIndex-GraphRAG/      LlamaIndex GraphRAG implementation
│   └── README.md             Setup and run instructions
└── eval-results/             Scored CSV/JSON output and summary tables


Each subproject README contains setup and run instructions for reproducing these results.

---


