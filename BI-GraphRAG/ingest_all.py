"""
ingest_all.py — One-shot ingestion for BI-GraphRAG
----------------------------------------------------
Run this once to load all CSVs + PDFs into Neo4j.

Usage:
    python ingest_all.py <path-to-data-folder>

Example:
    python ingest_all.py "E:\\Graph-RAG\\Fake_BI_data\\data"

The folder must contain:
    structured/   — CSV files (company, employees, orders, products …)
    unstructured/ — PDF / image files (QBRs, memos, meeting notes)
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


async def main(data_folder: Path):
    from core.config import settings
    from core.extract import extract_chunks_concurrent
    from core.ingest import IMAGE_EXTENSIONS, ingest_file
    from core.load import GraphLoader
    from ingest_structured import load_csv_triples

    t0 = time.perf_counter()

    # ── 0. Check ODL hybrid server before touching the graph ──────────────────
    import urllib.request, urllib.error
    try:
        urllib.request.urlopen("http://127.0.0.1:5002", timeout=2)
    except urllib.error.HTTPError:
        pass  # server responded with HTTP (e.g. 404) — it's running
    except Exception as e:
        print(f"\nERROR: ODL hybrid server not running at port 5002.")
        print(f"  Start it first:  opendataloader-pdf-hybrid --port 5002")
        print(f"  ({e})")
        sys.exit(1)
    print("✓ ODL hybrid server reachable at :5002\n")

    loader = GraphLoader()
    loader.clear_all()
    print("\n✓ Graph cleared\n")

    # ── 1. Unstructured PDFs / images ─────────────────────────────────────────
    unstructured_dir = data_folder / "unstructured"
    pdf_dir = unstructured_dir if unstructured_dir.exists() else data_folder
    allowed = {".pdf"} | IMAGE_EXTENSIONS
    pdf_files = sorted(f for f in pdf_dir.iterdir() if f.suffix.lower() in allowed)

    if not pdf_files:
        print("No PDF/image files found.")
    else:
        print(f"Processing {len(pdf_files)} PDF/image files concurrently …\n")

        pdf_triples_total = 0
        sem = asyncio.Semaphore(1)   # ODL hybrid server handles one PDF at a time

        async def process_one(pdf_path: Path) -> int:
            async with sem:
                t = time.perf_counter()
                try:
                    chunks = await asyncio.to_thread(
                        ingest_file, str(pdf_path), pdf_path.name,
                        settings.chunk_size, settings.chunk_overlap,
                    )
                except Exception as e:
                    # Fail loud for THIS file, but don't let one bad PDF (e.g. a
                    # docling backend crash on a complex table) discard every
                    # other PDF's already-completed work via asyncio.gather.
                    print(f"  ✗ {pdf_path.name:<45} FAILED: {e}")
                    return 0

                if not chunks:
                    print(f"  ⚠ {pdf_path.name} — no text extracted")
                    return 0

                triples = await extract_chunks_concurrent(chunks)
                if triples:
                    await asyncio.to_thread(loader.load_triples, triples)

                n = len(triples) if triples else 0
                elapsed = round((time.perf_counter() - t) * 1000)
                print(f"  ✓ {pdf_path.name:<45} {n:>4} triples  ({elapsed}ms)")
                return n

        results = await asyncio.gather(*[process_one(f) for f in pdf_files])
        pdf_triples_total = sum(results)
        n_failed = sum(1 for r in results if r == 0)
        print(f"\n✓ {pdf_triples_total} PDF triples loaded from {len(pdf_files)} files"
              + (f"  ({n_failed} file(s) contributed 0 triples — check ✗/⚠ lines above)" if n_failed else ""))

    # ── 2. Structured CSVs ────────────────────────────────────────────────────
    structured_dir = data_folder / "structured"
    if structured_dir.exists():
        print(f"\nLoading CSVs from {structured_dir} …")
        csv_triples = load_csv_triples(structured_dir)
        loader.load_triples(csv_triples)
        print(f"✓ {len(csv_triples)} CSV triples loaded")
    else:
        print("No structured/ subfolder found — skipping CSVs")

    loader.close()

    total_time = round(time.perf_counter() - t0)
    print(f"\n{'─'*55}")
    print(f"  Ingestion complete in {total_time}s")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.exists():
        print(f"ERROR: folder not found: {folder}")
        sys.exit(1)

    asyncio.run(main(folder))
