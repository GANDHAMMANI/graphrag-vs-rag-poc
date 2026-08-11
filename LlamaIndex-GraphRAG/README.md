# LlamaIndex GraphRAG — Comparison Project

Same dataset and same 30-question ground truth as BI-GraphRAG, but built with
LlamaIndex's `PropertyGraphIndex` instead of the hand-built pipeline. Goal:
find out whether the framework does better, worse, or about the same.

## Setup

1. Copy `.env.example` to `.env` and fill in:
   - `GROQ_API_KEY`
   - `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` — the **new** Neo4j
     instance, keep this separate from BI-GraphRAG's graph so the two don't mix.
2. Copy `.groq_keys.txt` over from BI-GraphRAG into this project's root if you
   want key rotation during RAGAs scoring (same file, same format, one key
   per line).
3. `venv\Scripts\pip install -r requirements.txt` (already done for the
   packages currently installed; rerun if you add anything).

## Run order

```
python ingest.py          # builds the graph in Neo4j via LlamaIndex — will take a while, LLM call per chunk
python eval/evaluate.py   # runs all 30 ground-truth questions, saves eval/results/llamaindex_eval.json
python eval/score.py      # scores with RAGAs, saves eval/results/llamaindex_scored.csv
```

Compare `llamaindex_scored.csv` numbers directly against BI-GraphRAG's
`graphrag_scored.csv` — same metrics, same questions, same judge model.

## What's different from BI-GraphRAG's hand-built pipeline

- Document loading, chunking, and entity/relationship extraction are all
  LlamaIndex's own logic (`SimpleDirectoryReader` + `SimpleLLMPathExtractor`),
  not our custom ODL Hybrid + Groq triple-extraction code.
- CSVs are loaded as plain documents here, not resolved via ID→name joins
  like BI-GraphRAG's CSV loader does. Worth keeping in mind if LlamaIndex
  underperforms on relational questions — that's an ingestion difference,
  not necessarily the framework itself.
- Retrieval uses LlamaIndex's built-in query engine (`as_query_engine`)
  rather than our manual seed-entity + hop-traversal + rerank pipeline.
