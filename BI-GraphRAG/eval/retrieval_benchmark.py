"""
retrieval_benchmark.py — Compare GraphRAG's three seed-finding strategies on
retrieval QUALITY (not latency, we already measured that): word-match only,
semantic-match only, and the current default (always run both, merge via RRF).

For each of the 30 ground-truth questions and each strategy, finds seed
entities, expands the graph from them (same as production), and saves the
retrieved facts as "contexts" — scored later with RAGAs context_precision /
context_recall against the ground truth to see which strategy actually
retrieves the right facts, not just which is fastest.

Usage:
    python eval/retrieval_benchmark.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retrieve import GraphRetriever

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

STRATEGIES = ["word_only", "semantic_only", "always_both"]


def get_seeds(retriever: GraphRetriever, tx, question: str, strategy: str, limit: int = 3):
    if strategy == "word_only":
        return retriever.find_seed_entities_by_words(tx, question, limit)
    elif strategy == "semantic_only":
        return retriever.find_seed_entities_by_meaning(tx, question, limit)
    else:  # always_both — current production default
        return retriever.find_seed_entities(tx, question, limit)


def run_strategy(retriever: GraphRetriever, questions: list[dict], strategy: str) -> list[dict]:
    rows = []
    with retriever.driver.session() as session:
        for q in questions:
            t0 = time.perf_counter()

            def _work(tx, q=q):
                seeds = get_seeds(retriever, tx, q["question"], strategy)
                all_facts = []
                seen = set()
                for seed in seeds:
                    facts = retriever.expand_context_structured(tx, seed, hops=2)
                    for f in facts:
                        key = (f["subject"], f["relationship"], f["object"])
                        if key not in seen:
                            seen.add(key)
                            all_facts.append(f)
                return seeds, all_facts

            seeds, facts = session.execute_read(_work)
            elapsed = round((time.perf_counter() - t0) * 1000)

            contexts = [f["fact_text"] for f in facts] or ["(no context retrieved)"]
            print(f"  [{q['id']:>2}] {strategy:<14} {elapsed:>5}ms  seeds={seeds}  facts={len(facts)}  {q['question'][:50]}")

            rows.append({
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "strategy": strategy,
                "seeds": seeds,
                "contexts": contexts,
                "ground_truth": q["ground_truth"],
            })
    return rows


def main():
    questions = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(questions)} ground-truth questions\n")

    retriever = GraphRetriever()

    all_results = {}
    for strategy in STRATEGIES:
        print(f"=== {strategy} ===")
        rows = run_strategy(retriever, questions, strategy)
        all_results[strategy] = rows
        out_path = RESULTS_DIR / f"retrieval_{strategy}.json"
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Saved to {out_path}\n")

    retriever.close()

    # Quick non-LLM sanity summary: how many questions got zero facts per strategy
    print("=== Quick summary (zero-context rate) ===")
    for strategy, rows in all_results.items():
        zero = sum(1 for r in rows if r["contexts"] == ["(no context retrieved)"])
        print(f"{strategy:<14} {zero}/{len(rows)} questions retrieved nothing")


if __name__ == "__main__":
    main()
