"""Orchestration: open and validate the PDF, route pages, assemble results."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import pymupdf

from .assembly import assemble
from .config import ExtractionConfig, ExtractionMode
from .errors import EncryptedPDFError, InvalidPDFError, PDFTooLargeError
from .layout import PageData
from .models import DocumentResult, PageResult
from .native import extract_native, image_coverage, native_word_count
from .ocr_engine import OCREngine
from .ocr_tables import extract_ocr

log = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"


class TableExtractor:
    """Reusable extractor. Holds one lazily loaded OCR engine across documents."""

    def __init__(self, config: ExtractionConfig | None = None, ocr_engine: OCREngine | None = None):
        self.config = config or ExtractionConfig()
        self._ocr = ocr_engine or OCREngine(
            min_score=self.config.ocr_min_score, max_side_len=self.config.ocr_max_side_len
        )

    # ------------------------------------------------------------------ input
    def _load(self, source: str | Path | bytes) -> tuple[bytes, str]:
        cfg = self.config
        limit = int(cfg.max_file_size_mb * 1024 * 1024)
        if isinstance(source, (bytes, bytearray)):
            data, name = bytes(source), "<bytes>"
        else:
            path = Path(source)
            if not path.is_file():
                raise InvalidPDFError(f"file not found: {path}")
            if path.stat().st_size > limit:
                raise PDFTooLargeError(f"{path} exceeds {cfg.max_file_size_mb} MB limit")
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise InvalidPDFError(f"cannot read {path}: {exc}") from exc
            name = str(path)
        if len(data) > limit:
            raise PDFTooLargeError(f"input exceeds {cfg.max_file_size_mb} MB limit")
        if PDF_MAGIC not in data[:1024]:
            raise InvalidPDFError(f"{name} is not a PDF (missing %PDF- header)")
        return data, name

    def _open(self, data: bytes, name: str) -> pymupdf.Document:
        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise InvalidPDFError(f"cannot parse {name}: {exc}") from exc
        if doc.needs_pass and not (self.config.password and doc.authenticate(self.config.password)):
            doc.close()
            raise EncryptedPDFError(f"{name} is encrypted; supply the correct password")
        if doc.page_count > self.config.max_pages:
            n = doc.page_count
            doc.close()
            raise PDFTooLargeError(f"{name} has {n} pages; limit is {self.config.max_pages}")
        return doc

    # ---------------------------------------------------------------- routing
    def _route(self, page: pymupdf.Page) -> str:
        mode = self.config.mode
        if mode is ExtractionMode.NATIVE:
            return "native"
        if mode is ExtractionMode.OCR:
            return "ocr"
        if native_word_count(page) >= self.config.min_native_words:
            return "native"
        if not page.get_images() and not page.get_drawings():
            return "skipped"  # blank page
        return "ocr"

    def _process_page(self, page: pymupdf.Page, page_no: int) -> tuple[PageData, str]:
        route = self._route(page)
        if route == "skipped":
            return PageData(page=page_no, width=page.rect.width, height=page.rect.height, tables=[], lines=[]), route
        if route == "native":
            data = extract_native(page, page_no, self.config)
            if (
                self.config.mode is ExtractionMode.AUTO
                and not data.tables
                and image_coverage(page) >= self.config.image_coverage_ocr_threshold
            ):
                log.info("page %d: text layer has no tables but page is mostly image; using OCR", page_no)
                return extract_ocr(page, page_no, self.config, self._ocr), "ocr"
            return data, "native"
        return extract_ocr(page, page_no, self.config, self._ocr), "ocr"

    # -------------------------------------------------------------------- run
    def extract(self, source: str | Path | bytes, pages: Iterable[int] | None = None) -> DocumentResult:
        """Extract every table. ``pages`` are 1-based page numbers (default all).

        Raises ``ExtractionError`` subclasses for document-level problems.
        Page-level failures are recorded in the result and do not abort the run.
        """
        started = time.perf_counter()
        data, name = self._load(source)
        digest = hashlib.sha256(data).hexdigest()
        doc = self._open(data, name)
        try:
            selected = sorted(set(pages)) if pages else list(range(1, doc.page_count + 1))
            bad = [p for p in selected if not 1 <= p <= doc.page_count]
            if bad:
                raise InvalidPDFError(f"page(s) out of range 1..{doc.page_count}: {bad}")

            page_data: list[PageData] = []
            page_results: list[PageResult] = []
            for page_no in selected:
                t0 = time.perf_counter()
                result = PageResult(page=page_no, method="error")
                try:
                    pdata, method = self._process_page(doc[page_no - 1], page_no)
                    page_data.append(pdata)
                    result.method = method
                    result.n_tables = len(pdata.tables)
                    result.warnings = pdata.warnings
                except Exception as exc:  # isolate page failures
                    log.exception("page %d failed", page_no)
                    result.error = f"{type(exc).__name__}: {exc}"
                result.duration_ms = (time.perf_counter() - t0) * 1000
                log.info("page %d: %s, %d table(s), %.0f ms", page_no, result.method, result.n_tables, result.duration_ms)
                page_results.append(result)

            # Headings for a selected page's first table may sit at the bottom of
            # an unselected previous page; read its text layer (never OCR).
            context: dict[int, PageData] = {}
            chosen = set(selected)
            for page_no in selected:
                prev_no = page_no - 1
                if prev_no >= 1 and prev_no not in chosen and self.config.mode is not ExtractionMode.OCR:
                    prev = doc[prev_no - 1]
                    if native_word_count(prev) >= self.config.min_native_words:
                        try:
                            context[prev_no] = extract_native(prev, prev_no, self.config)
                        except Exception:  # context is best effort only
                            log.debug("context page %d unavailable", prev_no, exc_info=True)

            return DocumentResult(
                source=name,
                sha256=digest,
                n_pages=doc.page_count,
                mode=self.config.mode.value,
                tables=assemble(page_data, self.config, context),
                pages=page_results,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        finally:
            doc.close()


def extract_tables(
    source: str | Path | bytes, config: ExtractionConfig | None = None, pages: Iterable[int] | None = None
) -> DocumentResult:
    """One-shot convenience wrapper around :class:`TableExtractor`."""
    return TableExtractor(config).extract(source, pages=pages)
