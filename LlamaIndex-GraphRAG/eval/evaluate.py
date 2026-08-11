"""
evaluate.py — Run the 30-question ground-truth set (copied from BI-GraphRAG)
through the LlamaIndex PropertyGraphIndex retriever and save answers for
RAGAs scoring, same output shape as BI-GraphRAG/eval/evaluate.py so the
existing score.py can be reused with a path change.

Usage:
    python eval/evaluate.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retrieve import LlamaIndexGraphRetriever

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    questions = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(questions)} ground-truth questions")

    retriever = LlamaIndexGraphRetriever()

    rows = []
    for q in questions:
        t0 = time.perf_counter()
        result = retriever.retrieve_with_citations(q["question"])
        elapsed = round((time.perf_counter() - t0) * 1000)
        print(f"  [{q['id']:>2}] {elapsed:>6}ms  {q['question'][:60]}")

        contexts = result["contexts"] or ["(no context retrieved)"]
        rows.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": result["answer"],
            "contexts": contexts,
            "ground_truth": q["ground_truth"],
        })

    out_path = RESULTS_DIR / "llamaindex_eval.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
