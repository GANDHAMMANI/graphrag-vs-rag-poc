"""
eval/score_retrieval.py — Score retrieval_benchmark.py's three seed strategies
with RAGAs context_precision / context_recall (no answer generation involved,
so Faithfulness/AnswerRelevancy don't apply here).

Usage:
    python eval/score_retrieval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextPrecision, ContextRecall
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import settings
from key_rotation import KeyRotator, RotatingChatGroq, load_keys

RESULTS_DIR = Path(__file__).parent / "results"
JUDGE_MODEL = "llama-3.3-70b-versatile"
STRATEGIES = ["word_only", "semantic_only", "always_both"]


def load_dataset(path: Path) -> Dataset:
    rows = json.loads(path.read_text(encoding="utf-8"))
    # Rows with no retrieved context shouldn't be scored as if "(no context
    # retrieved)" were a real fact — pass an empty context list instead so
    # RAGAs scores them as zero-recall/precision rather than judging the
    # placeholder string's similarity to the ground truth.
    contexts = [
        [] if r["contexts"] == ["(no context retrieved)"] else r["contexts"]
        for r in rows
    ]
    return Dataset.from_dict({
        "question":     [r["question"] for r in rows],
        "contexts":     contexts,
        "ground_truth": [r["ground_truth"] for r in rows],
    })


def main():
    keys = load_keys()
    print(f"Loaded {len(keys)} Groq API keys for judge rotation")
    rotator = KeyRotator(keys)
    judge_llm = LangchainLLMWrapper(RotatingChatGroq(rotator, model=JUDGE_MODEL, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.embed_model))

    metrics = [ContextPrecision(), ContextRecall()]
    run_config = RunConfig(max_workers=2, timeout=120, max_retries=6, max_wait=90)

    summary = {}
    for strategy in STRATEGIES:
        path = RESULTS_DIR / f"retrieval_{strategy}.json"
        if not path.exists():
            print(f"⚠ {path} not found — run eval/retrieval_benchmark.py first. Skipping {strategy}.")
            continue

        print(f"\n{'='*60}\nScoring {strategy}\n{'='*60}")
        dataset = load_dataset(path)

        result = evaluate(
            dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            raise_exceptions=False,
            run_config=run_config,
        )

        df = result.to_pandas()
        out_csv = RESULTS_DIR / f"retrieval_{strategy}_scored.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved per-question scores to {out_csv}")

        summary[strategy] = {
            "context_precision": round(df["context_precision"].mean(), 3),
            "context_recall":    round(df["context_recall"].mean(), 3),
        }

    if not summary:
        print("\nNo strategy results found — nothing scored.")
        return

    print(f"\n{'='*60}")
    print(f"{'Strategy':<18}{'Context Precision':>20}{'Context Recall':>20}")
    print(f"{'-'*60}")
    for strategy, scores in summary.items():
        print(f"{strategy:<18}{scores['context_precision']:>20}{scores['context_recall']:>20}")
    print(f"{'='*60}")

    (RESULTS_DIR / "retrieval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {RESULTS_DIR / 'retrieval_summary.json'}")


if __name__ == "__main__":
    main()