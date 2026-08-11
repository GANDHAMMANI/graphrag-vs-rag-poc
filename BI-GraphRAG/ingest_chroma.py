"""
ingest_chroma.py — Ingest all data into ChromaDB for Traditional RAG lane.

Run this AFTER ingest_all.py finishes (or independently).
Does NOT touch Neo4j — only ChromaDB.

Usage:
    python ingest_chroma.py "E:\\BI-GraphRAG\\data"
"""

import logging
import sys
import time
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def main(data_folder: Path):
    from chromadb.utils import embedding_functions
    import chromadb

    from core.config import settings
    from core.ingest import IMAGE_EXTENSIONS, ingest_file

    t0 = time.perf_counter()

    chroma_path = str(Path(__file__).parent / "chroma_db")
    client = chromadb.PersistentClient(path=chroma_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embed_model
    )

    # Wipe and recreate collection
    try:
        client.delete_collection("bi_chunks")
    except Exception:
        pass
    collection = client.create_collection(
        name="bi_chunks",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    print("\n✓ ChromaDB collection reset\n")

    docs, metas, ids = [], [], []

    def flush():
        if docs:
            collection.add(documents=docs, metadatas=metas, ids=ids)
            docs.clear(); metas.clear(); ids.clear()

    # ── 1. CSVs → human-readable text chunks ─────────────────────────────────
    structured_dir = data_folder / "structured"
    if structured_dir.exists():
        import pandas as pd

        def read(name):
            path = structured_dir / name
            return pd.read_csv(path) if path.exists() else pd.DataFrame()

        # Same ID→name lookups as ingest_structured.py (the GraphRAG lane) —
        # without this, foreign keys like region_id=region_1 get embedded as
        # opaque IDs instead of "North America", so questions like "who works
        # in North America" can't match against these rows at all.
        departments = read("departments.csv")
        regions     = read("regions.csv")
        products    = read("products.csv")
        employees   = read("employees.csv")
        customers   = read("customers.csv")

        id_to_name: dict[str, str] = {}
        for df_, id_col, name_col in [
            (departments, "id", "name"),
            (regions, "id", "name"),
            (products, "id", "name"),
            (employees, "id", "name"),
            (customers, "id", "name"),
        ]:
            if not df_.empty:
                id_to_name.update(dict(zip(df_[id_col], df_[name_col])))

        def resolve(col: str, val) -> str:
            if col.endswith("_id") and val in id_to_name:
                return f"{col}={val} ({id_to_name[val]})"
            return f"{col}={val}"

        csv_files = sorted(structured_dir.glob("*.csv"))
        print(f"Loading {len(csv_files)} CSV files into ChromaDB…")
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    text = f"{csv_path.stem}: " + " | ".join(
                        resolve(col, val) for col, val in row.items() if pd.notna(val)
                    )
                    docs.append(text)
                    metas.append({"source": csv_path.name, "type": "structured"})
                    ids.append(str(uuid.uuid4()))
                print(f"  ✓ {csv_path.name} — {len(df)} rows")
            except Exception as e:
                print(f"  ⚠ {csv_path.name} — {e}")

        flush()
        print(f"✓ CSV chunks loaded\n")

    # ── 2. PDFs → text chunks ─────────────────────────────────────────────────
    unstructured_dir = data_folder / "unstructured"
    pdf_dir = unstructured_dir if unstructured_dir.exists() else data_folder
    allowed = {".pdf"} | IMAGE_EXTENSIONS
    pdf_files = sorted(f for f in pdf_dir.iterdir() if f.suffix.lower() in allowed)

    if pdf_files:
        print(f"Loading {len(pdf_files)} PDF files into ChromaDB…")
        for pdf_path in pdf_files:
            try:
                chunks = ingest_file(
                    str(pdf_path), pdf_path.name,
                    settings.chunk_size, settings.chunk_overlap,
                )
                for chunk in chunks:
                    text = chunk.get("text", "").strip()
                    if not text:
                        continue
                    docs.append(text)
                    metas.append({
                        "source":      pdf_path.name,
                        "page_number": chunk.get("page_number") or 0,
                        "type":        "unstructured",
                    })
                    ids.append(str(uuid.uuid4()))
                print(f"  ✓ {pdf_path.name} — {len(chunks)} chunks")
            except Exception as e:
                print(f"  ⚠ {pdf_path.name} — {e}")

        flush()
        print(f"✓ PDF chunks loaded\n")

    total = round(time.perf_counter() - t0)
    total_docs = collection.count()
    print(f"{'─'*55}")
    print(f"  ChromaDB ingestion complete in {total}s")
    print(f"  Total documents: {total_docs}")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    folder = Path(sys.argv[1])
    if not folder.exists():
        print(f"ERROR: folder not found: {folder}")
        sys.exit(1)
    main(folder)
