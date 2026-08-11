"""
load_test.py — Concurrent load test + resource usage for GraphRAG retrieval.
-------------------------------------------------------------------------------
Fires N requests at CONCURRENCY simultaneous workers against retrieve_with_citations,
for both seed-lookup backends (Neo4j-native vector index vs separate Chroma store),
while sampling this process's own CPU/memory in the background.

Note: Neo4j here is AuraDB (cloud-hosted) — its own server resource usage isn't
visible to us. This measures our side only: the retrieval/API process, including
the local embedding model, cross-encoder, and ChromaDB (all in-process).

Usage:
    python eval/load_test.py [--concurrency 10] [--total 40]
"""

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psutil

from core.retrieve import GraphRetriever

QUESTIONS = [
    "What did John Lewis discuss?",
    "Who is the CEO?",
    "Who works in North America?",
    "What products fall under Security?",
    "Trevor Campos promotion",
    "What did Courtney Keller discuss?",
    "Who is the Head of Sales?",
    "What did Patricia Marshall discuss?",
    "Who was promoted to Senior CS Manager?",
    "What products were part of the Marketing featured deals?",
]


async def _one_request(retriever: GraphRetriever, question: str, vector_backend: str) -> float:
    t0 = time.perf_counter()
    await asyncio.to_thread(_retrieve, retriever, question, vector_backend)
    return (time.perf_counter() - t0) * 1000


def _retrieve(retriever: GraphRetriever, question: str, vector_backend: str):
    with retriever.driver.session() as session:
        seeds = session.execute_read(lambda tx: retriever.find_seed_entities(tx, question, 3, vector_backend))
        if not seeds:
            return
        for seed in seeds:
            session.execute_read(retriever.expand_context_structured, seed, 2)


async def run_load(retriever: GraphRetriever, backend: str, total: int, concurrency: int) -> dict:
    proc = psutil.Process()
    proc.cpu_percent()  # prime the counter (first call always returns 0)

    cpu_samples, mem_samples = [], []
    stop = False

    async def sampler():
        while not stop:
            cpu_samples.append(proc.cpu_percent())
            mem_samples.append(proc.memory_info().rss / (1024 * 1024))  # MB
            await asyncio.sleep(0.2)

    sampler_task = asyncio.create_task(sampler())

    sem = asyncio.Semaphore(concurrency)
    latencies = []

    async def worker(i: int):
        async with sem:
            q = QUESTIONS[i % len(QUESTIONS)]
            ms = await _one_request(retriever, q, backend)
            latencies.append(ms)

    t0 = time.perf_counter()
    await asyncio.gather(*[worker(i) for i in range(total)])
    wall = time.perf_counter() - t0

    stop = True
    await sampler_task

    latencies.sort()
    def pct(p): return latencies[min(int(len(latencies) * p), len(latencies) - 1)]

    return {
        "backend": backend,
        "total_requests": total,
        "concurrency": concurrency,
        "wall_seconds": round(wall, 2),
        "throughput_rps": round(total / wall, 2),
        "latency_ms": {
            "min": round(min(latencies)),
            "p50": round(pct(0.50)),
            "p95": round(pct(0.95)),
            "max": round(max(latencies)),
        },
        "process_cpu_pct": {
            "avg": round(statistics.mean(cpu_samples), 1) if cpu_samples else 0,
            "max": round(max(cpu_samples), 1) if cpu_samples else 0,
        },
        "process_memory_mb": {
            "avg": round(statistics.mean(mem_samples)) if mem_samples else 0,
            "max": round(max(mem_samples)) if mem_samples else 0,
        },
    }


def print_result(r: dict):
    print(f"\n{'='*60}")
    print(f"Backend: {r['backend']}  |  concurrency={r['concurrency']}  total={r['total_requests']}")
    print(f"{'='*60}")
    print(f"Wall time:        {r['wall_seconds']}s")
    print(f"Throughput:       {r['throughput_rps']} req/s")
    print(f"Latency (ms):     min={r['latency_ms']['min']}  p50={r['latency_ms']['p50']}  p95={r['latency_ms']['p95']}  max={r['latency_ms']['max']}")
    print(f"Process CPU %:    avg={r['process_cpu_pct']['avg']}  max={r['process_cpu_pct']['max']}")
    print(f"Process memory MB: avg={r['process_memory_mb']['avg']}  max={r['process_memory_mb']['max']}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--total", type=int, default=40)
    args = parser.parse_args()

    retriever = GraphRetriever()

    # Warm up: load embedding model + cross-encoder + prime both backends once,
    # so the first real request in each run isn't skewed by cold-start cost.
    print("Warming up models…")
    with retriever.driver.session() as session:
        session.execute_read(lambda tx: retriever.find_seed_entities(tx, "warmup", 1, "neo4j"))
        session.execute_read(lambda tx: retriever.find_seed_entities(tx, "warmup", 1, "chroma"))

    results = []
    for backend in ["neo4j", "chroma"]:
        r = await run_load(retriever, backend, args.total, args.concurrency)
        print_result(r)
        results.append(r)

    retriever.close()

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    print(f"{'Backend':<10}{'Throughput':>14}{'p50 (ms)':>12}{'p95 (ms)':>12}{'CPU avg%':>10}{'Mem avg MB':>12}")
    for r in results:
        print(f"{r['backend']:<10}{r['throughput_rps']:>14}{r['latency_ms']['p50']:>12}{r['latency_ms']['p95']:>12}{r['process_cpu_pct']['avg']:>10}{r['process_memory_mb']['avg']:>12}")


if __name__ == "__main__":
    asyncio.run(main())
