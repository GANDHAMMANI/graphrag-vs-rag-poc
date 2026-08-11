"""
score.py — RAGAs scoring for the LlamaIndex GraphRAG lane.

Reads eval/results/llamaindex_eval.json (from evaluate.py) and scores it
with the same four metrics used for BI-GraphRAG, so the numbers are directly
comparable: faithfulness, answer relevancy, context precision, context recall.

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
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import settings
from key_rotation import KeyRotator, RotatingChatGroq, load_keys

RESULTS_DIR = Path(__file__).parent / "results"
JUDGE_MODEL = "llama-3.3-70b-versatile"


def main():
    keys = load_keys()
    print(f"Loaded {len(keys)} Groq API keys for judge rotation")
    rotator = KeyRotator(keys)
    judge_llm = LangchainLLMWrapper(RotatingChatGroq(rotator, model=JUDGE_MODEL, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.embed_model))

    metrics = [Faithfulness(), AnswerRelevancy(strictness=1), ContextPrecision(), ContextRecall()]
    run_config = RunConfig(max_workers=2, timeout=120, max_retries=6, max_wait=90)

    rows = json.loads((RESULTS_DIR / "llamaindex_eval.json").read_text(encoding="utf-8"))
    dataset = Dataset.from_dict({
        "question":     [r["question"] for r in rows],
        "answer":       [r["answer"] for r in rows],
        "contexts":     [r["contexts"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })

    result = evaluate(dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings,
                       raise_exceptions=False, run_config=run_config)

    df = result.to_pandas()
    df.to_csv(RESULTS_DIR / "llamaindex_scored.csv", index=False)

    print(f"\n{'='*50}")
    print(f"LlamaIndex GraphRAG — RAGAs results")
    print(f"{'='*50}")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"{metric:<20}{round(df[metric].mean(), 3)}")


if __name__ == "__main__":
    main()
