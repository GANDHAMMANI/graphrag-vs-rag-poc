"""
evaluate.py — RAGAs evaluation for GraphRAG vs Traditional RAG (ChromaDB)
--------------------------------------------------------------------------
1. Loads ground_truth.json (30 questions: relational / narrative / mixed)
2. Runs both retrievers directly (no API server needed) and generates
   answers using the same prompts/models as api.py
3. Scores each lane with RAGAs: faithfulness, answer_relevancy,
   context_precision, context_recall
4. Prints a comparison table and saves per-question results to CSV

Usage:
    python eval/evaluate.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from groq import AsyncGroq

from core.config import settings
from core.retrieve import GraphRetriever
from core.chroma_retrieve import ChromaRetriever

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BI_SYSTEM_PROMPT = """You are a business intelligence analyst. Answer using ONLY the facts provided — no inference, no assumptions.

Rules:
- Write in natural, fluent prose — not a keyword dump. Synthesize the facts into readable sentences.
- Report ONLY what is explicitly in the facts. Never say "can be inferred" or "can be assumed".
- If numbers are listed, report them exactly as given. Do not compute missing values.
- Cover ALL facts provided — do not omit items from lists (products, employees, orders).
- If something is not in the facts, say so in one short sentence. No guessing.
- 2-4 sentences or a short paragraph. No bullet points unless listing 4+ items."""

CHROMA_SYSTEM_PROMPT = """You are a business intelligence assistant using traditional document retrieval.
Answer using ONLY the text chunks provided — no inference, no assumptions.
- Report only what is explicitly in the chunks.
- If the answer spans multiple chunks, combine them naturally.
- If something is not in the chunks, say "not found in documents" in one sentence.
- Keep answers concise."""


async def run_graphrag_lane(questions: list[dict], retriever: GraphRetriever, client: AsyncGroq) -> list[dict]:
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        result = await asyncio.to_thread(
            retriever.retrieve_with_citations, q["question"], 2, settings.max_context_facts
        )
        context = result["context"]
        contexts = [c.strip().lstrip("- ").strip() for c in context.split("\n") if c.strip()] if context else []

        if not context:
            answer = "No relevant information found in the knowledge graph."
        else:
            response = await client.chat.completions.create(
                model=settings.answer_model,
                messages=[
                    {"role": "system", "content": BI_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Facts from knowledge graph:\n{context}\n\nQuestion: {q['question']}"},
                ],
                temperature=0.1,
            )
            answer = response.choices[0].message.content

        elapsed = round((time.perf_counter() - t0) * 1000)
        print(f"  [{q['id']:>2}] GraphRAG  {elapsed:>5}ms  {q['question'][:60]}")
        rows.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": answer,
            "contexts": contexts if contexts else ["(no context retrieved)"],
            "ground_truth": q["ground_truth"],
        })
    return rows


async def run_chroma_lane(questions: list[dict], retriever: ChromaRetriever, client: AsyncGroq) -> list[dict]:
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        result = await asyncio.to_thread(retriever.retrieve, q["question"], 5)
        context = result["context"]
        contexts = [c.strip() for c in context.split("\n\n---\n\n") if c.strip()] if context else []

        if not context:
            answer = "No relevant chunks found in the document store."
        else:
            response = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": CHROMA_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Chunks:\n{context}\n\nQuestion: {q['question']}"},
                ],
                temperature=0.1,
            )
            answer = response.choices[0].message.content

        elapsed = round((time.perf_counter() - t0) * 1000)
        print(f"  [{q['id']:>2}] ChromaDB  {elapsed:>5}ms  {q['question'][:60]}")
        rows.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": answer,
            "contexts": contexts if contexts else ["(no context retrieved)"],
            "ground_truth": q["ground_truth"],
        })
    return rows


async def main():
    questions = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(questions)} ground-truth questions\n")

    graph_retriever = GraphRetriever()
    chroma_retriever = ChromaRetriever()
    client = AsyncGroq(api_key=settings.groq_api_key)

    print("Running GraphRAG lane…")
    graph_rows = await run_graphrag_lane(questions, graph_retriever, client)

    print("\nRunning ChromaDB (Traditional RAG) lane…")
    chroma_rows = await run_chroma_lane(questions, chroma_retriever, client)

    graph_retriever.close()

    (RESULTS_DIR / "graphrag_eval.json").write_text(json.dumps(graph_rows, indent=2), encoding="utf-8")
    (RESULTS_DIR / "chroma_eval.json").write_text(json.dumps(chroma_rows, indent=2), encoding="utf-8")

    print(f"\nSaved generation outputs to {RESULTS_DIR}/graphrag_eval.json and chroma_eval.json")
    print("Next: run  python eval/score.py  to compute RAGAs metrics.")


if __name__ == "__main__":
    asyncio.run(main())
