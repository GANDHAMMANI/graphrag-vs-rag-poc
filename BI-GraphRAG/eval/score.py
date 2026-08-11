"""
score.py — Compute RAGAs metrics for GraphRAG vs Traditional RAG (ChromaDB)
------------------------------------------------------------------------------
Reads eval/results/{graphrag,chroma}_eval.json (produced by evaluate.py),
scores each with RAGAs (faithfulness, answer_relevancy, context_precision,
context_recall) using Groq as the judge LLM, and prints a comparison table.

Usage:
    python eval/score.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import settings
from key_rotation import KeyRotator, RotatingChatGroq, load_keys

RESULTS_DIR = Path(__file__).parent / "results"

# Stronger judge now that key rotation gives us the whole pool's quota, not
# just one key's 100k tokens/day.
JUDGE_MODEL = "llama-3.3-70b-versatile"


def load_dataset(path: Path) -> Dataset:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return Dataset.from_dict({
        "question":     [r["question"] for r in rows],
        "answer":       [r["answer"] for r in rows],
        "contexts":     [r["contexts"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    }), rows


def main():
    keys = load_keys()
    print(f"Loaded {len(keys)} Groq API keys for judge rotation")
    rotator = KeyRotator(keys)
    judge_llm = LangchainLLMWrapper(RotatingChatGroq(rotator, model=JUDGE_MODEL, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.embed_model))

    # strictness=1 avoids AnswerRelevancy requesting n>1 completions per call,
    # which Groq rejects outright ("'n': number must be at most 1").
    metrics = [
        Faithfulness(),
        AnswerRelevancy(strictness=1),
        ContextPrecision(),
        ContextRecall(),
    ]

    # Throttle concurrency + add generous retries/backoff so a burst of judge
    # calls doesn't slam Groq's rate limiter all at once.
    run_config = RunConfig(max_workers=2, timeout=120, max_retries=6, max_wait=90)

    summary = {}
    for lane, fname in [("GraphRAG", "graphrag_eval.json"), ("Traditional RAG (ChromaDB)", "chroma_eval.json")]:
        print(f"\n{'='*60}\nScoring {lane}\n{'='*60}")
        dataset, rows = load_dataset(RESULTS_DIR / fname)

        result = evaluate(
            dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            raise_exceptions=False,
            run_config=run_config,
        )

        df = result.to_pandas()
        out_csv = RESULTS_DIR / f"{lane.split()[0].lower()}_scored.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved per-question scores to {out_csv}")

        summary[lane] = {
            "faithfulness":      round(df["faithfulness"].mean(), 3),
            "answer_relevancy":  round(df["answer_relevancy"].mean(), 3),
            "context_precision": round(df["context_precision"].mean(), 3),
            "context_recall":    round(df["context_recall"].mean(), 3),
        }

    print(f"\n{'='*70}")
    print(f"{'Metric':<20}{'GraphRAG':>15}{'Traditional RAG':>20}{'Winner':>15}")
    print(f"{'-'*70}")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        g = summary["GraphRAG"][metric]
        c = summary["Traditional RAG (ChromaDB)"][metric]
        winner = "GraphRAG" if g > c else ("ChromaDB" if c > g else "tie")
        print(f"{metric:<20}{g:>15}{c:>20}{winner:>15}")
    print(f"{'='*70}")

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
