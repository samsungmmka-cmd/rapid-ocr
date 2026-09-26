"""DocumentDataExtractor: full content of ANY PDF (and images) in one call.

Per PDF page:
  * digital page (usable text layer)  -> exact native text; tables from the
    existing TableExtractor (native mode, unchanged logic incl. titles and
    cross-page merging); fields from text and from PDF form widgets.
  * scanned / image-only page, or a text layer that is garbled (broken font
    encoding)                           -> ImageDataExtractor (OCR ensemble).
  * blank page                         -> empty page.
Images are passed straight to ImageDataExtractor.

All coordinates are pixels of the page rendered at ``ImageDataConfig.pdf_dpi``
(original pixels for images), so native and OCR pages are comparable.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import time
from dataclasses import replace
from pathlib import Path
from statistics import median

import numpy as np
import pymupdf

from .config import ExtractionConfig, ExtractionMode
from .errors import EncryptedPDFError, InvalidPDFError, PDFTooLargeError
from .image_data import ImageDataConfig, ImageDataExtractor, ImageDataResult, PageData, render_pdf_page
from .image_layout import Block, Item, KeyValue, inline_kv, pair_kvs, text_blocks, xy_cut
from .native import _text_lines, native_word_count
from .ocr_models import OCRLine, OCRModels
from .pipeline import TableExtractor
from .reader import detect_type

log = logging.getLogger(__name__)

GARBLED_RATIO = 0.2  # share of unreadable glyphs that makes a text layer unusable


def _garbled(page: pymupdf.Page) -> bool:
    text = "".join(page.get_text().split())
    if not text:
        return False
    bad = sum(1 for ch in text if ch == "�" or 0xE000 <= ord(ch) <= 0xF8FF or (ord(ch) < 32))
    return bad / len(text) > GARBLED_RATIO


class DocumentDataExtractor:
    """Extract all content (text, headings, fields, tables) from PDFs and images."""

    def __init__(
        self,
        config: ImageDataConfig | None = None,
        table_config: ExtractionConfig | None = None,
        models: OCRModels | None = None,
    ):
        self.config = config or ImageDataConfig()
        self.images = ImageDataExtractor(self.config, models=models)
        base = table_config or ExtractionConfig(password=self.config.password, max_pages=self.config.max_pages,
                                                max_file_size_mb=self.config.max_file_size_mb)
        # Tables for digital pages only; scanned pages are handled by the OCR path.
        self.tables = TableExtractor(replace(base, mode=ExtractionMode.NATIVE))

    def extract(self, source: str | Path | bytes, pages: list[int] | None = None) -> ImageDataResult:
        started = time.perf_counter()
        if isinstance(source, (bytes, bytearray)):
            data, name = bytes(source), "<bytes>"
            if len(data) > self.config.max_file_size_mb * 1024 * 1024:
                raise PDFTooLargeError(f"input exceeds {self.config.max_file_size_mb} MB limit")
        else:
            path = Path(source)
            if not path.is_file():
                raise InvalidPDFError(f"file not found: {path}")
            if path.stat().st_size > self.config.max_file_size_mb * 1024 * 1024:
                raise PDFTooLargeError(f"{path} exceeds {self.config.max_file_size_mb} MB limit")
            data, name = path.read_bytes(), str(path)
        if detect_type(data[:1024]) != "pdf":
            result = self.images.extract(data, pages=pages)
            result.source = name
            return result

        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise InvalidPDFError(f"cannot parse {name}: {exc}") from exc
        try:
            if doc.needs_pass and not (self.config.password and doc.authenticate(self.config.password)):
                raise EncryptedPDFError(f"{name} is encrypted; supply the correct password")
            if doc.page_count > self.config.max_pages:
                raise PDFTooLargeError(f"{name} has {doc.page_count} pages; limit is {self.config.max_pages}")
            selected = sorted(set(pages)) if pages else list(range(1, doc.page_count + 1))
            bad = [p for p in selected if not 1 <= p <= doc.page_count]
            if bad:
                raise InvalidPDFError(f"page(s) out of range 1..{doc.page_count}: {bad}")

            routes = {p: self._route(doc[p - 1]) for p in selected}
            native_pages = [p for p, r in routes.items() if r == "native"]
            tables = self.tables.extract(data, pages=native_pages).tables if native_pages else []

            results: list[PageData] = []
            for p in selected:
                t0 = time.perf_counter()
                try:
                    page = self._page(doc[p - 1], p, routes[p], tables)
                except Exception as exc:  # isolate page failures
                    log.exception("page %d failed", p)
                    page = PageData(p, 0, 0, [], [], [], [], {"method": routes[p]}, error=f"{type(exc).__name__}: {exc}")
                page.duration_ms = (time.perf_counter() - t0) * 1000
                log.info("page %d: %s, %d blocks, %d fields, %d tables", p, routes[p], len(page.blocks), len(page.key_values), len(page.tables))
                results.append(page)
        finally:
            doc.close()

        index = 0
        for pg in results:
            for t in pg.tables:
                t.index = index
                index += 1
        return ImageDataResult(
            source=name,
            sha256=hashlib.sha256(data).hexdigest(),
            input_type="pdf",
            file_format="PDF",
            pages=results,
            accuracy=self.config.accuracy,
            models_unavailable=self.images.models.unavailable,
            review_threshold=self.config.review_threshold,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    # ------------------------------------------------------------ routing
    def _route(self, page: pymupdf.Page) -> str:
        if native_word_count(page) >= 5 and not _garbled(page):
            return "native"
        if page.get_images() or page.get_drawings() or page.get_text().strip():
            return "ocr"
        return "blank"

    def _page(self, page: pymupdf.Page, page_no: int, route: str, doc_tables) -> PageData:
        scale = self.config.pdf_dpi / 72
        w, h = int(page.rect.width * scale), int(page.rect.height * scale)
        if route == "blank":
            return PageData(page_no, w, h, [], [], [], [], {"method": "blank", "level": "good", "issues": [], "notes": ["Blank page."]})
        if route == "ocr":
            result = self.images.extract_page(render_pdf_page(page, self.config.pdf_dpi, self.config.max_pixels), page_no)
            result.quality["method"] = "ocr"
            if _garbled(page):
                result.quality.setdefault("notes", []).append("The PDF text layer is garbled (broken font encoding); the page was OCR'd instead.")
            return result
        return self._native_page(page, page_no, scale, w, h, doc_tables)

    # ------------------------------------------------------------- native
    def _native_page(self, page, page_no: int, scale: float, w: int, h: int, doc_tables) -> PageData:
        page_tables = [t for t in doc_tables if t.page == page_no]
        quality = {"method": "native", "level": "good", "issues": [], "notes": [], "enhancements": [], "ocr_confidence": None}
        if _degenerate(page_tables):
            # Layout boxes were detected as one big table: analyse the page
            # layout instead, with the exact native text.
            quality["notes"].append("Page layout uses boxed regions; structure recovered by layout analysis on native text.")
            return self._native_layout_page(page, page_no, scale, w, h, quality)

        # Continuation parts of multi-page tables have no bbox on this page:
        # exclude text inside any native table region on the page.
        from .native import extract_native

        regions = [t.bbox for t in extract_native(page, page_no, ExtractionConfig()).tables]
        lines = _lines(page, regions, scale)
        tables = []
        for t in page_tables:
            t = copy.deepcopy(t)
            t.bbox = tuple(v * scale for v in t.bbox)
            tables.append(t)

        kvs, paired = pair_kvs(lines, w)
        for ln in lines:
            if ln.id not in paired and (kv := inline_kv(ln)):
                kvs.append(kv)
                paired.add(ln.id)
        kvs += _widget_kvs(page, scale)

        lh = median(ln.height for ln in lines) if lines else 12 * scale
        items = [Item(t.bbox, "table", i) for i, t in enumerate(tables)]
        items += [Item(kv.bbox, "kv", i) for i, kv in enumerate(kvs)]
        items += [Item(ln.bbox, "line", ln) for ln in lines if ln.id not in paired]
        blocks: list[Block] = []
        for leaf in xy_cut(items, lh) if items else []:
            leaf_lines = [it.ref for it in leaf if it.kind == "line"]
            leaf_blocks = text_blocks(leaf_lines, lh, lh) if leaf_lines else []
            for it in leaf:
                if it.kind == "table":
                    leaf_blocks.append(Block("table", "", it.bbox, 1.0, ref=it.ref))
                elif it.kind == "kv":
                    kv = kvs[it.ref]
                    leaf_blocks.append(Block("key_value", f"{kv.key}: {kv.value}", kv.bbox, 1.0, kv.line_ids, ref=it.ref))
            blocks += sorted(leaf_blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

        return PageData(page_no, w, h, blocks, kvs, tables, lines, quality)

    def _native_layout_page(self, page, page_no: int, scale: float, w: int, h: int, quality: dict) -> PageData:
        lines = _lines(page, [], scale)
        img = render_pdf_page(page, self.config.pdf_dpi, self.config.max_pixels)
        if img.shape[1] != w:  # pixel cap reduced the resolution: rescale text coordinates
            f = img.shape[1] / w
            for ln in lines:
                ln.poly = ln.poly * f
        char_h = 0.6 * median(ln.height for ln in lines) if lines else 10 * scale
        blocks, kvs, tables = self.images._layout(img, lines, char_h, allow_reread=False)
        for t in tables:
            t.page = t.pages[0] = page_no
            t.method = t.method.replace("image-", "native-layout-")
        widget_kvs = _widget_kvs(page, scale)
        for kv in widget_kvs:
            blocks.append(Block("key_value", f"{kv.key}: {kv.value}", kv.bbox, 1.0, ref=len(kvs)))
            kvs.append(kv)
        return PageData(page_no, w, h, blocks, kvs, tables, lines, quality)


def _lines(page, exclude, scale: float) -> list[OCRLine]:
    out = []
    for i, tl in enumerate(_text_lines(page, exclude)):
        x0, y0, x1, y1 = (v * scale for v in tl.bbox)
        out.append(OCRLine(id=i, poly=np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float), text=tl.text, score=1.0, model="native"))
    return out


def _widget_kvs(page, scale: float) -> list[KeyValue]:
    """Values typed into PDF form fields (they are not part of the page text)."""
    kvs = []
    for widget in page.widgets() or []:
        value = widget.field_value
        if widget.field_name and value not in (None, "", "Off"):
            kvs.append(KeyValue(widget.field_name, str(value), 1.0, tuple(v * scale for v in widget.rect), "form_field"))
    return kvs


def _degenerate(tables) -> bool:
    """A table enclosing another table, or a header that is one long merged
    cell, means page layout boxes were read as a table."""
    def encloses(a, b):
        return a[0] <= b[0] + 1 and a[1] <= b[1] + 1 and a[2] >= b[2] - 1 and a[3] >= b[3] - 1

    for t in tables:
        if any(o is not t and encloses(t.bbox, o.bbox) for o in tables):
            return True
        filled = [c for c in t.header if c]
        if len(t.header) >= 2 and len(filled) == 1 and len(filled[0]) > 60:
            return True
    return False


def extract_content(source: str | Path | bytes, config: ImageDataConfig | None = None, pages: list[int] | None = None) -> ImageDataResult:
    """One-shot helper: full content of a PDF or image."""
    return DocumentDataExtractor(config).extract(source, pages=pages)
