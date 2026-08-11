"""
Stage 1: Entity & Relationship Extraction
-------------------------------------------
Takes raw text -> returns a list of (subject, relationship, object) triples
using an LLM (Groq) with a strict JSON schema in the prompt.

Why this matters: the quality of your whole GraphRAG system depends on
how clean these triples are. Vague prompts = messy, duplicate, or
hallucinated entities.

Provides both:
  extract_triples()           — synchronous, one chunk at a time
  extract_chunks_concurrent() — async, fires up to N Groq calls in parallel
"""

import asyncio
import json
import logging

from groq import AsyncGroq, Groq

from .config import settings

logger = logging.getLogger(__name__)


_sync_client = Groq(api_key=settings.groq_api_key)


EXTRACTION_SYSTEM_PROMPT = """Extract entity-relationship triples from the text.

Return ONLY valid JSON, no markdown:
{"triples":[{"subject":"...","relationship":"UPPER_SNAKE_CASE","object":"...","subject_type":"...","object_type":"..."}]}

Rules:
- subject_type/object_type: Person | Organization | Document | Location | Topic | Event | Concept | Skill | Certification | Measurement | Finding
- relationship: UPPER_SNAKE_CASE verb (AUTHORED, CONDUCTED, WORKS_AT, IS_A, HAS_ROLE, HAS_SKILL,
  HAS_CERTIFICATION, ACHIEVED, LOCATED_AT, RECORDED, IDENTIFIED, HAS_VALUE, HAS_STATUS,
  HAS_FEATURE, CONTAINS, PART_OF, RELATED_TO, DISCUSSED, ASKED_ABOUT, PLANS, REQUESTED,
  PROVIDED_FEEDBACK_ON, MET_WITH, WILL_SEND, DECIDED, MENTIONED, HIGHLIGHTED, RAISED_CONCERN …)
- Canonical names: one consistent spelling per entity throughout
- Only explicitly stated or strongly implied facts — no invention
- Always add IS_A or HAS_ROLE for roles/titles
- Extract ALL numerical measurements as triples — e.g. "2321 CFM's" → Blower Door Test RECORDED 2321 CFMs
- Extract certifications — e.g. "John Doe, CCHI" → John Doe HAS_CERTIFICATION CCHI
- Extract findings/defects — e.g. "air leakage at rim joist" → Rim Joist IDENTIFIED Air Leakage
- For meeting notes extract discussion content fully:
  "discussed current usage" → John Lewis DISCUSSED Current Usage
  "asked about roadmap" → John Lewis ASKED_ABOUT Product Roadmap
  "plans to decide before Q-end" → Barrera-Martin PLANS Renewal Decision
  "Sherri will send a proposal" → Sherri Frazier WILL_SEND Proposal
- For QBRs/memos extract highlights, decisions, concerns, and action items as triples
- CRITICAL — named subjects only: NEVER use "Author", "Author of text", "The author", "Writer", or
  any placeholder. Skip triples where subject or object is unnamed.
- CRITICAL — no sentence fragments as entities: subject and object must each be a short named
  thing (a person, company, product, document, role, or specific term) — never a clause or
  summary phrase copied from the text. Reduce commentary to its subject.
  BAD:  "Sales team's results for Q1 2026" as an entity
  GOOD: Sales HAS_RESULTS "Q1 2026"  (or similar, split into a proper subject/object pair)
  BAD:  "Continue reporting into the broader Engineering organization for 2025 planning"
  GOOD: Engineering REPORTS_INTO "Broader Engineering Organization"
- Empty result: {"triples":[]}"""


_GENERIC_SUBJECTS = {"author", "author of text", "the author", "writer", "the writer", "narrator"}

# Backstop for the prompt rule above — the LLM doesn't always follow it, so also
# reject anything that reads like a sentence fragment rather than a named entity:
# too many words, or contains a lowercase filler word that a real entity name
# wouldn't (e.g. "team's results for", "continue reporting into").
_MAX_ENTITY_WORDS = 5
_FRAGMENT_MARKERS = {
    "results for", "reporting into", "adoption across", "coordination with",
    "continue to", "planning for", "team's", "'s results", "'s q",
}

def _is_generic(name: str) -> bool:
    return name.strip().lower() in _GENERIC_SUBJECTS

def _is_sentence_fragment(name: str) -> bool:
    stripped = name.strip()
    if len(stripped.split()) > _MAX_ENTITY_WORDS:
        return True
    lowered = stripped.lower()
    return any(marker in lowered for marker in _FRAGMENT_MARKERS)

def _filter_generic(triples: list[dict]) -> list[dict]:
    """Drop triples whose subject or object is a placeholder or a narrative
    sentence fragment mistaken for a named entity."""
    kept = []
    for t in triples:
        subj, obj = t.get("subject", ""), t.get("object", "")
        if _is_generic(subj) or _is_generic(obj):
            continue
        if _is_sentence_fragment(subj) or _is_sentence_fragment(obj):
            continue
        kept.append(t)
    return kept


# ── Rule-based measurement extraction ────────────────────────────────────────
# LLMs reliably miss numerical measurements in structured report formats
# (e.g. "10.1 Air-Flow\nComments: Serviceable\n2321 CFM's").
# Regex handles these deterministically and complements LLM entity extraction.

import re as _re

_MEASUREMENT_RULES: list[tuple] = [
    # (pattern, subject, relationship, object_template, subject_type, object_type)
    # Blower door airflow — matches "2321 CFM" / "2,321 CFMs"
    (r'([\d,]+(?:\.\d+)?)\s*CFMs?\b', "Blower Door Test", "RECORDED", "{} CFMs", "Event", "Measurement"),
    # Air changes per hour
    (r'([\d]+(?:\.\d+)?)\s*ACH', "Air Leakage Test", "RECORDED", "{} ACH50", "Event", "Measurement"),
    # Insulation R-value — "R-20", "R20"
    (r'\bR-?(\d+)\b', "Insulation", "HAS_VALUE", "R-{}", "Topic", "Measurement"),
    # Pressure in Pa / KPA
    (r'([\d]+(?:\.\d+)?)\s*(?:KPA|kPa|Pa)\b', "Pressure Test", "RECORDED", "{} Pa", "Event", "Measurement"),
    # Temperature — "120 degrees F" / "35°C"
    (r'([\d]+(?:\.\d+)?)\s*(?:degrees?\s*[FCfc]|°[FCfc])', "Temperature", "RECORDED", "{}", "Measurement", "Measurement"),
    # BTU ratings
    (r'([\d,]+(?:\.\d+)?)\s*BTU', "HVAC System", "RATED", "{} BTU", "Topic", "Measurement"),
]


def _extract_measurements(text: str) -> list[dict]:
    """
    Rule-based pass that extracts numerical measurements the LLM commonly skips.
    Returns triples WITHOUT provenance (caller adds it via _attach_provenance).
    """
    triples = []
    seen = set()
    for pattern, subject, rel, obj_tmpl, s_type, o_type in _MEASUREMENT_RULES:
        for m in _re.finditer(pattern, text, _re.IGNORECASE):
            raw_val = m.group(1).replace(",", "")
            obj = obj_tmpl.format(raw_val)
            key = (subject, rel, obj)
            if key in seen:
                continue
            seen.add(key)
            triples.append({
                "subject": subject,
                "relationship": rel,
                "object": obj,
                "subject_type": s_type,
                "object_type": o_type,
            })
    return triples

def _attach_provenance(triples: list[dict], page_number, source, bbox=None, section=None) -> list[dict]:
    """Stamp page_number, source, bbox, and section onto every triple in-place."""
    for t in triples:
        if page_number is not None:
            t["page_number"] = page_number
        if source is not None:
            t["source"] = source
        if bbox is not None:
            t["bbox"] = list(bbox) if not isinstance(bbox, list) else bbox
        if section is not None:
            t["section"] = section
    return triples


# ── Synchronous (backward-compatible) ────────────────────────────────────────

def extract_triples(
    text: str,
    model: str | None = None,
    page_number: int | None = None,
    source: str | None = None,
    bbox=None,
    section: str | None = None,
) -> list[dict]:
    """
    Send one text chunk to Groq synchronously and return parsed triples.
    page_number / source / section are attached to every returned triple when provided.
    """
    model = model or settings.extraction_model
    try:
        response = _sync_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        triples = _filter_generic(json.loads(raw).get("triples", []))
    except json.JSONDecodeError as exc:
        logger.error("JSON decode failed: %s", exc)
        return []
    except Exception as exc:
        logger.error("extract_triples failed: %s", exc)
        return []

    triples += _extract_measurements(text)
    return _attach_provenance(triples, page_number, source, bbox, section)


# ── Asynchronous / concurrent ─────────────────────────────────────────────────

async def _extract_one_async(
    chunk: dict,
    client: AsyncGroq,
    semaphore: asyncio.Semaphore,
    model: str,
) -> list[dict]:
    """Extract triples from a single chunk, respecting the shared semaphore."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": chunk["text"]},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            triples = _filter_generic(json.loads(raw).get("triples", []))
        except json.JSONDecodeError as exc:
            logger.error("JSON decode failed on chunk (page %s): %s", chunk.get("page_number"), exc)
            return []
        except Exception as exc:
            logger.error("Async extraction failed on chunk (page %s): %s", chunk.get("page_number"), exc)
            return []

        triples += _extract_measurements(chunk["text"])
        return _attach_provenance(
            triples,
            chunk.get("page_number"),
            chunk.get("source"),
            chunk.get("bbox"),
            chunk.get("section"),
        )


async def extract_chunks_concurrent(
    chunks: list[dict],
    max_concurrent: int | None = None,
    model: str | None = None,
) -> list[dict]:
    """
    Fire up to max_concurrent Groq extraction calls in parallel using asyncio.

    This is the production-speed path: a 40-chunk document that takes ~80s
    sequentially finishes in ~16s with max_concurrent=5.

    Parameters
    ----------
    chunks          : output of ingest_pdf() — each dict has text, page_number, source
    max_concurrent  : cap on simultaneous Groq requests (default: settings value)
    model           : Groq model name (default: settings.extraction_model)

    Returns
    -------
    Flat list of all triples extracted from all chunks, with provenance attached.
    Chunks that fail are logged and skipped — they don't abort the whole batch.
    """
    if not chunks:
        return []

    max_concurrent = max_concurrent or settings.max_concurrent_extractions
    model = model or settings.extraction_model

    semaphore = asyncio.Semaphore(max_concurrent)
    client = AsyncGroq(api_key=settings.groq_api_key)

    logger.info(
        "Extracting from %d chunks, max_concurrent=%d", len(chunks), max_concurrent
    )

    tasks = [
        _extract_one_async(chunk, client, semaphore, model)
        for chunk in chunks
    ]
    # return_exceptions=True means one failed chunk won't cancel the rest.
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_triples: list[dict] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.error("Chunk %d raised an exception: %s", i, result)
        else:
            all_triples.extend(result)

    logger.info("Extracted %d triples from %d chunks", len(all_triples), len(chunks))
    return all_triples


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter text: ")
    triples = extract_triples(text)
    print(json.dumps(triples, indent=2))
