"""
ingest.py — PDF and image ingestion for GraphRAG
-------------------------------------------------
PDF extraction: OpenDataLoader PDF Hybrid ONLY (docling-fast server required on port 5002).
Image files: Tesseract OCR via pytesseract.

Requires:
  pip install -U opendataloader-pdf
  Start hybrid server:  opendataloader-pdf-hybrid --port 5002

Every element returned carries {type, text, page_number, bbox}.
Every chunk returned carries {text, page_number, bbox, type, section, source}.

Public API
----------
    ingest_file(source, source_name="", chunk_size=1200, overlap=150) -> list[dict]
    ingest_pdf(source, source_name="", chunk_size=1200, overlap=150)  -> list[dict]

    source: file path (str | Path) OR file-like object (BytesIO, UploadFile …)
"""

import io as _io
import logging
import os
import tempfile
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

logger = logging.getLogger(__name__)

# ── Element type normalisation ────────────────────────────────────────────────
_TYPE_MAP = {
    "title": "heading", "heading": "heading",
    "header": "heading", "sectionheader": "heading",
    "table": "table",
    "text": "text", "narrativetext": "text",
    "listitem": "text", "list": "text",
    "footer": "text", "uncategorizedtext": "text",
}

def _norm_type(raw: str) -> str:
    return _TYPE_MAP.get(raw.lower().replace(" ", ""), "text")


# ── 1. OpenDataLoader PDF Hybrid ─────────────────────────────────────────────

_ODL_HYBRID_URL = "http://127.0.0.1:5002"

def _load_via_opendataloader(source) -> list[dict]:
    """
    OpenDataLoader PDF Hybrid — #1 in benchmarks (0.907 overall score).
    Requires docling-fast server running on port 5002.

    Start server: opendataloader-pdf-hybrid --port 5002
    Install:      pip install -U opendataloader-pdf
    """
    try:
        import opendataloader_pdf  # type: ignore
    except ImportError:
        raise ImportError(
            "opendataloader-pdf not installed — run: pip install -U opendataloader-pdf"
        )

    import json
    import shutil
    import urllib.request

    # Verify hybrid server is reachable — any HTTP response means it's up
    import urllib.error
    try:
        urllib.request.urlopen(_ODL_HYBRID_URL, timeout=2)
    except urllib.error.HTTPError:
        pass  # got an HTTP response (e.g. 404) — server is running
    except Exception as e:
        raise RuntimeError(
            f"ODL hybrid server not reachable at {_ODL_HYBRID_URL}. "
            f"Run:  opendataloader-pdf-hybrid --port 5002  Error: {e}"
        )

    # Write file-like objects to a temp file
    _tmp_pdf = None
    if hasattr(source, "read"):
        data = source.read()
        _tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        _tmp_pdf.write(data); _tmp_pdf.flush(); _tmp_pdf.close()
        pdf_path = _tmp_pdf.name
    else:
        pdf_path = str(source)

    output_dir = tempfile.mkdtemp()
    try:
        opendataloader_pdf.convert(
            input_path=[pdf_path],
            output_dir=output_dir,
            format="json",
            reading_order="xycut",
            table_method="cluster",
            hybrid="docling-fast",
            hybrid_url=_ODL_HYBRID_URL,
        )

        json_files = list(Path(output_dir).glob("*.json"))
        if not json_files:
            raise RuntimeError("ODL produced no JSON output")

        with open(json_files[0], encoding="utf-8") as f:
            raw = json.load(f)

        # ODL JSON structure: top-level dict with metadata + "kids" list of elements
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            items = raw.get("kids") or raw.get("elements") or []
        else:
            items = []

        if items:
            logger.info("ODL JSON sample item type=%s value=%r", type(items[0]).__name__, str(items[0])[:300])

        def _parse_kid(item) -> list[dict]:
            """Recursively parse an ODL kid element, flattening nested kids."""
            if not isinstance(item, dict):
                return []
            el_type = str(item.get("type", "text")).lower()
            # Image elements from scanned pages — no text content, skip
            if el_type in ("image", "figure"):
                return []
            text = str(item.get("content") or item.get("text") or "").strip()
            page = item.get("page number") or item.get("page_number") or item.get("page")
            try:
                page = int(page) if page is not None else None
            except (TypeError, ValueError):
                page = None
            bbox_raw = item.get("bounding box") or item.get("bbox")
            bbox = [float(v) for v in bbox_raw] if isinstance(bbox_raw, list) and len(bbox_raw) == 4 else None

            results = []
            if text:
                results.append({
                    "type":        _norm_type(el_type),
                    "text":        text,
                    "page_number": page,
                    "bbox":        bbox,
                })
            # Recurse into nested kids (ODL sometimes nests elements)
            for child in item.get("kids") or []:
                results.extend(_parse_kid(child))
            return results

        elements: list[dict] = []
        for item in items:
            elements.extend(_parse_kid(item))

        if not elements:
            logger.warning(
                "ODL returned no text elements from %s (got %d kids, all non-text). "
                "Scanned/image-only pages need force_ocr mode.",
                Path(pdf_path).name, len(items)
            )
            return []

        logger.info("ODL hybrid: %d elements", len(elements))
        return elements

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        if _tmp_pdf:
            os.unlink(_tmp_pdf.name)


# ── 2. unstructured (fallback) ────────────────────────────────────────────────

def _load_via_unstructured(source) -> list[dict]:
    """
    Use the `unstructured` library with strategy="hi_res" for PDFs that
    contain embedded images, charts, or complex tables.

    Install: pip install "unstructured[pdf]"
    Optional for best results: pip install "unstructured[pdf,local-inference]"
    """
    try:
        from unstructured.partition.pdf import partition_pdf  # type: ignore
    except ImportError:
        raise ImportError("unstructured not installed — pip install 'unstructured[pdf]'")

    def _extract(path: str) -> list:
        return partition_pdf(
            path,
            strategy="hi_res",
            infer_table_structure=True,
            include_metadata=True,
        )

    if hasattr(source, "read"):
        data = source.read()
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            tmp.write(data); tmp.flush(); tmp.close()
            raw = _extract(tmp.name)
        finally:
            os.unlink(tmp.name)
    else:
        raw = _extract(str(source))

    elements: list[dict] = []
    for el in raw:
        text = str(el).strip()
        if not text:
            continue
        cat = el.category if hasattr(el, "category") else "Text"
        meta = el.metadata if hasattr(el, "metadata") else None
        page = getattr(meta, "page_number", None) if meta else None
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None

        bbox = None
        coords = getattr(meta, "coordinates", None) if meta else None
        if coords and hasattr(coords, "points"):
            pts = coords.points
            try:
                x0 = min(p[0] for p in pts)
                y0 = min(p[1] for p in pts)
                x1 = max(p[0] for p in pts)
                y1 = max(p[1] for p in pts)
                bbox = [float(x0), float(y0), float(x1), float(y1)]
            except Exception:
                pass

        elements.append({
            "type": _norm_type(str(cat)),
            "text": text,
            "page_number": page,
            "bbox": bbox,
        })

    if not elements:
        raise RuntimeError("unstructured returned no elements")

    logger.info("unstructured: %d elements", len(elements))
    return elements


# ── 2. PyMuPDF hybrid (text blocks + embedded image OCR) ─────────────────────

def _load_via_pymupdf(source) -> list[dict]:
    """
    PyMuPDF with:
    - Per-block text extraction (each block has its own bbox)
    - Font-size heading detection
    - Table extraction via find_tables()
    - Embedded image OCR via pytesseract (if available)
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed — pip install PyMuPDF")

    try:
        import pytesseract  # type: ignore
        from PIL import Image as _PIL_Image  # type: ignore
        _HAS_OCR = True
    except ImportError:
        _HAS_OCR = False

    if hasattr(source, "read"):
        data = source.read()
        doc = fitz.open(stream=data, filetype="pdf")
    else:
        doc = fitz.open(str(source))

    elements: list[dict] = []

    for page_num, page in enumerate(doc, start=1):
        # ── Tables ───────────────────────────────────────────────────────────
        try:
            for table in page.find_tables().tables:
                try:
                    import pandas as pd
                    df = table.to_pandas()
                    if not df.empty:
                        bbox = list(table.bbox) if hasattr(table, "bbox") else None
                        elements.append({
                            "type": "table",
                            "text": df.to_markdown(index=False),
                            "page_number": page_num,
                            "bbox": bbox,
                        })
                except Exception:
                    pass
        except Exception:
            pass

        # ── Text + image blocks ───────────────────────────────────────────────
        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", [])

        # Collect font sizes to detect headings via median comparison
        sizes: list[float] = []
        for blk in blocks:
            if blk.get("type") != 0:
                continue
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        sizes.append(span.get("size", 12.0))

        median_size = sorted(sizes)[len(sizes) // 2] if sizes else 12.0

        # Count how many spans are bold on this page to avoid flagging
        # everything as a heading in an all-bold document.
        total_spans, bold_spans = 0, 0
        for blk in blocks:
            if blk.get("type") != 0:
                continue
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        total_spans += 1
                        flags = span.get("flags", 0)
                        font = span.get("font", "").lower()
                        if (flags & (1 << 4)) or "bold" in font:
                            bold_spans += 1
        # If >60 % of page text is bold, don't use boldness as heading signal
        # (the whole page is bold — can't distinguish headings that way).
        bold_meaningful = total_spans > 0 and (bold_spans / total_spans) < 0.6

        for blk in blocks:
            blk_type = blk.get("type")

            # Image block → OCR if available
            if blk_type == 1 and _HAS_OCR:
                try:
                    xref_list = blk.get("image")
                    xref = xref_list[0] if xref_list else None
                    if xref:
                        img_info = doc.extract_image(xref)
                        img = _PIL_Image.open(_io.BytesIO(img_info["image"]))
                        ocr_text = pytesseract.image_to_string(img, lang="eng").strip()
                        if ocr_text:
                            elements.append({
                                "type": "text",
                                "text": ocr_text,
                                "page_number": page_num,
                                "bbox": list(blk.get("bbox")) if blk.get("bbox") else None,
                            })
                except Exception:
                    pass
                continue

            if blk_type != 0:
                continue

            # Text block — collect all spans; detect heading by font size OR bold
            block_text = ""
            max_size = 0.0
            any_bold = False
            all_bold = True
            span_count = 0
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    t = span.get("text", "").strip()
                    if not t:
                        continue
                    block_text += t + " "
                    sz = span.get("size", 12.0)
                    max_size = max(max_size, sz)
                    flags = span.get("flags", 0)
                    font = span.get("font", "").lower()
                    is_span_bold = bool(flags & (1 << 4)) or "bold" in font
                    any_bold = any_bold or is_span_bold
                    all_bold = all_bold and is_span_bold
                    span_count += 1

            block_text = block_text.strip()
            if not block_text:
                continue

            # A block is a heading if:
            #   • its font is larger than the page median (classic heading), OR
            #   • it is bold and bold text is meaningful on this page (not everything bold),
            #     AND it is short enough to plausibly be a heading (≤ 120 chars).
            is_large_font = max_size > median_size * 1.15
            is_bold_heading = bold_meaningful and any_bold and len(block_text) <= 120
            el_type = "heading" if (is_large_font or is_bold_heading) else "text"

            bbox = list(blk["bbox"]) if blk.get("bbox") else None
            elements.append({
                "type": el_type,
                "text": block_text,
                "page_number": page_num,
                "bbox": bbox,
            })

    doc.close()
    logger.info("PyMuPDF hybrid: %d elements", len(elements))
    return [e for e in elements if e["text"].strip()]


# ── 3. pdfplumber ─────────────────────────────────────────────────────────────

def _load_via_pdfplumber(source) -> list[dict]:
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        raise ImportError("pdfplumber not installed — pip install pdfplumber")

    if hasattr(source, "read"):
        pdf = pdfplumber.open(source)
    else:
        pdf = pdfplumber.open(str(source))

    elements: list[dict] = []
    with pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                if table:
                    rows = [
                        " | ".join(str(cell or "").strip() for cell in row)
                        for row in table
                    ]
                    text = "\n".join(rows)
                    if text.strip():
                        elements.append({
                            "type": "table",
                            "text": text,
                            "page_number": page_num,
                            "bbox": None,
                        })

            text = page.extract_text() or ""
            if text.strip():
                elements.append({
                    "type": "text",
                    "text": text.strip(),
                    "page_number": page_num,
                    "bbox": None,
                })

    logger.info("pdfplumber: %d elements", len(elements))
    return [e for e in elements if e["text"].strip()]


# ── Image file loader (JPG / PNG / TIFF / BMP / WEBP) ────────────────────────

def _load_via_image_ocr(source) -> list[dict]:
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except ImportError:
        raise ImportError(
            "Image OCR requires: pip install Pillow pytesseract\n"
            "Also install the Tesseract binary."
        )

    if hasattr(source, "read"):
        img = Image.open(source)
    else:
        img = Image.open(str(source))

    text = pytesseract.image_to_string(img, lang="eng").strip()
    if not text:
        raise RuntimeError("pytesseract returned no text from image.")

    logger.info("Image OCR: extracted %d chars", len(text))
    return [{"type": "text", "text": text, "page_number": 1, "bbox": None}]


# ── Waterfall loader ──────────────────────────────────────────────────────────

def load_elements(source) -> list[dict]:
    """
    Load PDF elements using OpenDataLoader PDF Hybrid only.
    Raises RuntimeError if the docling-fast server is not running on port 5002.
    """
    return _load_via_opendataloader(source)


def load_image_elements(source) -> list[dict]:
    """Entry point for raw image files (JPG, PNG, TIFF, BMP, WEBP)."""
    try:
        return _load_via_image_ocr(source)
    except ImportError as exc:
        logger.error("Image OCR not available: %s", exc)
    except Exception as exc:
        logger.error("Image OCR failed: %s", exc)
    return []


# ── Structure-aware chunking ──────────────────────────────────────────────────

# Minimum vertical gap (in PDF points) between two text blocks that indicates
# they belong to different sections of the document.
# Typical line spacing ≈ 14–18pt; a section break is usually 30–60pt.
_SECTION_GAP_PTS = 40


def _bbox_section_break(prev_bbox, curr_bbox) -> bool:
    """
    Return True when two consecutive elements are far enough apart vertically
    to be considered different document sections (e.g. Skills vs Experience).
    Only fires when both elements have concrete bboxes and are on the same page.
    """
    if prev_bbox is None or curr_bbox is None:
        return False
    # bbox = [x0, y0, x1, y1]; y0 is the top of the block
    gap = curr_bbox[1] - prev_bbox[3]  # curr top - prev bottom
    return gap > _SECTION_GAP_PTS


def _make_chunks_from_buffer(
    buffer: list[dict],
    section: str | None,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """
    Merge buffered elements and split with a sliding window.

    Key improvement over the old _flush_section:
    - Tracks (page_number, bbox) per character offset so that sub-chunks that
      start in different parts of the document get the correct bbox, not just
      the first element's bbox.
    """
    if not buffer:
        return []

    combined = "\n".join(e["text"] for e in buffer)

    # Build per-character-position index: (start_offset, page_number, bbox)
    offsets: list[tuple[int, int | None, list | None]] = []
    pos = 0
    for el in buffer:
        offsets.append((pos, el["page_number"], el["bbox"]))
        pos += len(el["text"]) + 1  # +1 for the \n

    def attrs_at(char_pos: int) -> tuple[int | None, list | None]:
        pg, bx = offsets[0][1], offsets[0][2]
        for offset, p, b in offsets:
            if char_pos >= offset:
                pg, bx = p, b
            else:
                break
        return pg, bx

    if len(combined) <= chunk_size:
        pg, bx = attrs_at(0)
        return [{
            "text": combined,
            "page_number": pg,
            "bbox": bx,
            "type": "text",
            "section": section,
        }]

    chunks = []
    start = 0
    while start < len(combined):
        piece = combined[start: start + chunk_size].strip()
        if piece:
            pg, bx = attrs_at(start)
            chunks.append({
                "text": piece,
                "page_number": pg,
                "bbox": bx,
                "type": "text",
                "section": section,
            })
        start += chunk_size - overlap
    return chunks


def build_chunks(
    elements: list[dict],
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[dict]:
    """
    Structure-aware chunking with accurate per-chunk bbox attribution.

    Strategy:
    - Headings open new sections; text accumulates until the next heading.
    - Tables are always their own chunk.
    - When no headings are detected (flat documents like resumes), we break at
      significant vertical gaps between text blocks so Skills, Experience, and
      Projects each become separate chunks with their own correct bbox.
    - Long sections are split with a sliding window that tracks bbox per
      character offset — sub-chunks get the bbox of the element they start in,
      not the bbox of the first element in the buffer.
    """
    if not elements:
        return []

    has_headings = any(e["type"] == "heading" for e in elements)
    chunks: list[dict] = []
    current_section: str | None = None
    buffer: list[dict] = []

    def flush():
        nonlocal buffer
        if buffer:
            chunks.extend(_make_chunks_from_buffer(buffer, current_section, chunk_size, overlap))
            buffer = []

    for el in elements:
        if el["type"] == "table":
            flush()
            chunks.append({
                "text": el["text"],
                "page_number": el["page_number"],
                "bbox": el["bbox"],
                "type": "table",
                "section": current_section,
            })

        elif el["type"] == "heading" and has_headings:
            flush()
            current_section = el["text"].strip()
            chunks.append({
                "text": el["text"].strip(),
                "page_number": el["page_number"],
                "bbox": el["bbox"],
                "type": "heading",
                "section": current_section,
            })

        else:
            if not el["text"].strip():
                continue

            # Without explicit headings, break the buffer at significant
            # vertical gaps so each document section gets its own bbox.
            if (
                buffer
                and not has_headings
                and (
                    el.get("page_number") != buffer[-1].get("page_number")
                    or _bbox_section_break(buffer[-1].get("bbox"), el.get("bbox"))
                )
            ):
                flush()

            buffer.append(el)

    flush()
    chunks = [c for c in chunks if c["text"].strip()]
    logger.info("build_chunks: %d chunks from %d elements", len(chunks), len(elements))
    return chunks


# ── Public entry points ───────────────────────────────────────────────────────

def _finalise(elements: list[dict], source_name: str, chunk_size: int, overlap: int) -> list[dict]:
    if not elements:
        logger.warning("No elements extracted from '%s'", source_name)
        return []
    chunks = build_chunks(elements, chunk_size=chunk_size, overlap=overlap)
    for chunk in chunks:
        chunk["source"] = source_name
    return chunks


def ingest_pdf(
    source,
    source_name: str = "",
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[dict]:
    """
    Load a PDF using OpenDataLoader PDF Hybrid and return retrieval-ready chunks.
    Raises RuntimeError if the docling-fast server is not running at port 5002.
    """
    elements = load_elements(source)
    return _finalise(elements, source_name, chunk_size, overlap)


def ingest_image(
    source,
    source_name: str = "",
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[dict]:
    """
    OCR a single image file (JPG, PNG, TIFF, BMP, WEBP) and return chunks.
    Requires Pillow + pytesseract.
    """
    try:
        elements = load_image_elements(source)
    except Exception as exc:
        logger.error("Unexpected error loading image '%s': %s", source_name, exc)
        return []
    return _finalise(elements, source_name, chunk_size, overlap)


def ingest_file(
    source,
    source_name: str = "",
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[dict]:
    """
    Universal entry point — routes to ingest_pdf or ingest_image based on
    file extension.
    """
    ext = Path(source_name).suffix.lower() if source_name else ""
    if not ext and not hasattr(source, "read"):
        ext = Path(str(source)).suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return ingest_image(source, source_name, chunk_size, overlap)
    return ingest_pdf(source, source_name, chunk_size, overlap)
