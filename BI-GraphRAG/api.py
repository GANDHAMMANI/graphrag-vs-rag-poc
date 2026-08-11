"""
api.py — GraphRAG Business Intelligence API
--------------------------------------------
Ingestion:
  python ingest_all.py <data-folder>     → Neo4j (GraphRAG)
  python ingest_chroma.py <data-folder>  → ChromaDB (Traditional RAG)

Endpoints:
  POST /ask       GraphRAG answer (Neo4j)
  POST /ask-both  Side-by-side: GraphRAG vs Traditional RAG in parallel
  DELETE /graph   Wipe Neo4j graph
  GET  /health    Liveness check
  GET  /          Serve UI

Run:
    uvicorn api:app --host 0.0.0.0 --port 8001 --reload
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from groq import AsyncGroq
from pydantic import BaseModel

from core.config import settings
from core.load import GraphLoader
from core.retrieve import GraphRetriever
from core.chroma_retrieve import ChromaRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _preload_models():
    from core.embeddings import get_model
    from core.rerank import _load_cross_encoder
    logger.info("Preloading embedding model…")
    get_model()
    logger.info("Preloading cross-encoder…")
    _load_cross_encoder(settings.rerank_model)
    logger.info("Models ready.")


_retriever: "GraphRetriever | None" = None
_chroma: "ChromaRetriever | None" = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever, _chroma
    await asyncio.to_thread(_preload_models)
    _retriever = GraphRetriever()
    _chroma    = ChromaRetriever()
    logger.info("Neo4j + ChromaDB retrievers initialised.")
    yield
    if _retriever:
        _retriever.close()


app = FastAPI(title="BI-GraphRAG API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])




# ── /ask ──────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    hops: int = 2


BI_SYSTEM_PROMPT = """You are a business intelligence analyst. Answer using ONLY the facts provided — no inference, no assumptions.

Rules:
- Write in natural, fluent prose — not a keyword dump. Synthesize the facts into readable sentences.
- Report ONLY what is explicitly in the facts. Never say "can be inferred" or "can be assumed".
- If numbers are listed, report them exactly as given. Do not compute missing values.
- Cover ALL facts provided — do not omit items from lists (products, employees, orders).
- If something is not in the facts, say so in one short sentence. No guessing.
- 2-4 sentences or a short paragraph. No bullet points unless listing 4+ items."""


@app.post("/ask")
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    t_start = time.perf_counter()

    result = await asyncio.to_thread(
        _retriever.retrieve_with_citations, req.question, req.hops, settings.max_context_facts
    )

    t_retrieved = time.perf_counter()

    context   = result["context"]
    citations = result["citations"][:settings.max_citations]
    sources   = result["sources"]

    if not context:
        return {
            "answer": "No relevant information found in the knowledge graph.",
            "citations": [], "sources": [],
            "latency": {"retrieval_ms": round((t_retrieved - t_start) * 1000), "generation_ms": 0, "total_ms": 0},
            "tokens":  {"prompt": 0, "completion": 0, "total": 0},
        }

    prompt = f"Facts from knowledge graph:\n{context}\n\nQuestion: {req.question}"

    client   = AsyncGroq(api_key=settings.groq_api_key)
    response = await client.chat.completions.create(
        model=settings.answer_model,
        messages=[
            {"role": "system", "content": BI_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
    )

    t_end  = time.perf_counter()
    answer = response.choices[0].message.content
    usage  = response.usage

    return {
        "answer": answer,
        "citations": citations,
        "sources":   sources,
        "latency": {
            "retrieval_ms":  round((t_retrieved - t_start) * 1000),
            "generation_ms": round((t_end - t_retrieved) * 1000),
            "total_ms":      round((t_end - t_start) * 1000),
        },
        "tokens": {
            "prompt":     usage.prompt_tokens     if usage else 0,
            "completion": usage.completion_tokens if usage else 0,
            "total":      usage.total_tokens      if usage else 0,
        },
    }


# ── /ask-both ─────────────────────────────────────────────────────────────────

CHROMA_SYSTEM_PROMPT = """You are a business intelligence assistant using traditional document retrieval.
Answer using ONLY the text chunks provided — no inference, no assumptions.
- Report only what is explicitly in the chunks.
- If the answer spans multiple chunks, combine them naturally.
- If something is not in the chunks, say "not found in documents" in one sentence.
- Keep answers concise."""


async def _run_graphrag(question: str, hops: int) -> dict:
    t_start = time.perf_counter()
    result = await asyncio.to_thread(
        _retriever.retrieve_with_citations, question, hops, settings.max_context_facts
    )
    t_retrieved = time.perf_counter()

    context   = result["context"]
    citations = result["citations"][:settings.max_citations]

    if not context:
        return {"answer": "No relevant information found in the knowledge graph.",
                "citations": [], "latency": {}, "tokens": {}}

    client = AsyncGroq(api_key=settings.groq_api_key)
    response = await client.chat.completions.create(
        model=settings.answer_model,
        messages=[
            {"role": "system", "content": BI_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Facts:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.1,
    )
    t_end = time.perf_counter()
    usage = response.usage
    return {
        "answer":    response.choices[0].message.content,
        "citations": citations,
        "latency": {
            "retrieval_ms":  round((t_retrieved - t_start) * 1000),
            "generation_ms": round((t_end - t_retrieved) * 1000),
            "total_ms":      round((t_end - t_start) * 1000),
        },
        "tokens": {
            "prompt":     usage.prompt_tokens     if usage else 0,
            "completion": usage.completion_tokens if usage else 0,
            "total":      usage.total_tokens      if usage else 0,
        },
    }


async def _run_chroma(question: str) -> dict:
    t_start = time.perf_counter()
    result = await asyncio.to_thread(_chroma.retrieve, question, 5)
    t_retrieved = time.perf_counter()

    context   = result["context"]
    citations = result["citations"]

    if not context:
        return {"answer": "No relevant chunks found in the document store.",
                "citations": [], "latency": {}, "tokens": {}}

    client = AsyncGroq(api_key=settings.groq_api_key)
    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": CHROMA_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Chunks:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.1,
    )
    t_end = time.perf_counter()
    usage = response.usage
    return {
        "answer":    response.choices[0].message.content,
        "citations": citations,
        "latency": {
            "retrieval_ms":  round((t_retrieved - t_start) * 1000),
            "generation_ms": round((t_end - t_retrieved) * 1000),
            "total_ms":      round((t_end - t_start) * 1000),
        },
        "tokens": {
            "prompt":     usage.prompt_tokens     if usage else 0,
            "completion": usage.completion_tokens if usage else 0,
            "total":      usage.total_tokens      if usage else 0,
        },
    }


@app.post("/ask-both")
async def ask_both(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    graphrag_result, chroma_result = await asyncio.gather(
        _run_graphrag(req.question, req.hops),
        _run_chroma(req.question),
    )
    return {"graphrag": graphrag_result, "traditional_rag": chroma_result}


# ── Utility endpoints ─────────────────────────────────────────────────────────

@app.delete("/graph")
async def delete_graph():
    def _wipe():
        loader = GraphLoader()
        loader.clear_all()
        loader.close()
    await asyncio.to_thread(_wipe)
    return {"message": "Graph cleared."}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def ui():
    return FileResponse(Path(__file__).parent / "ui.html", media_type="text/html")
