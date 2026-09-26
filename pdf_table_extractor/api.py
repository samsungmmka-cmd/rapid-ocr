"""FastAPI service: extract content or tables from PDFs and images.

Run:   uvicorn pdf_table_extractor.api:app --host 0.0.0.0 --port 8000
Docs:  http://localhost:8000/docs

Environment:
  API_KEY               if set, requests must send header ``X-API-Key``
  MAX_UPLOAD_MB         upload size limit (default 50)
  MAX_CONCURRENT_JOBS   parallel extractions (default 2); others wait
  JOB_WAIT_SECONDS      how long a request waits for a free slot (default 120)
  MAX_PAGES             pages per document (default 100)
  MAX_PIXELS            pixels per decoded image / rendered page (default 60,000,000)
"""

from __future__ import annotations

import hmac
import logging
import os
import re
import threading
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from . import __version__
from .config import ExtractionConfig
from .document_data import DocumentDataExtractor
from .errors import EncryptedPDFError, ExtractionError, OCRUnavailableError, PDFTooLargeError
from .image_data import ImageDataConfig
from .image_reader import ImageTableExtractor, _OrientationOCR
from .ocr_engine import OCREngine
from .ocr_models import OCRModels
from .pipeline import TableExtractor
from .reader import UnsupportedFileError, detect_type

log = logging.getLogger(__name__)

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
JOB_WAIT = float(os.getenv("JOB_WAIT_SECONDS", "120"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "100"))
MAX_PIXELS = int(os.getenv("MAX_PIXELS", "60000000"))
_MULTIPART_OVERHEAD = 64 * 1024
_slots = threading.BoundedSemaphore(MAX_JOBS)

class BodyLimitMiddleware:
    """Reject oversized requests BEFORE the multipart parser spools them to
    memory/disk: checks Content-Length and counts streamed bytes (chunked)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        limit = int(MAX_UPLOAD_MB * 1024 * 1024) + _MULTIPART_OVERHEAD
        headers = dict(scope.get("headers") or [])
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = 0
        if declared > limit:
            return await _too_large(send)
        state = {"received": 0, "exceeded": False, "sent": False}

        async def limited_receive():
            if state["exceeded"]:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                state["received"] += len(message.get("body", b""))
                if state["received"] > limit:
                    # Stop reading; the parser fails and we answer 413 below.
                    state["exceeded"] = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            if state["exceeded"]:
                if not state["sent"]:
                    state["sent"] = True
                    await _too_large(send)
                return
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not state["exceeded"]:
                raise
        if state["exceeded"] and not state["sent"]:
            await _too_large(send)


async def _too_large(send) -> None:
    body = f'{{"error":"too_large","detail":"request exceeds {MAX_UPLOAD_MB:g} MB"}}'.encode()
    await send({"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


def _safe_name(filename: str | None) -> str:
    """Client filenames are untrusted: keep a short base name without paths or
    control characters."""
    name = (filename or "upload").replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()[:128]
    return name or "upload"


app = FastAPI(
    title="Document Extraction API",
    version=__version__,
    description="Extract full content (text, headings, fields, tables) or tables only from PDFs and images.",
)
app.add_middleware(BodyLimitMiddleware)


# ------------------------------------------------------- shared OCR models
# Models are loaded once per process and shared by all requests (thread-safe).
@lru_cache(maxsize=None)
def _models(accuracy: str) -> OCRModels:
    medium = accuracy != "fast"
    return OCRModels(use_medium=medium, use_english=medium)


@lru_cache(maxsize=1)
def _table_engines() -> tuple[OCREngine, _OrientationOCR]:
    return OCREngine(), _OrientationOCR()


def _check_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("API_KEY")
    if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def _parse_pages(pages: str | None) -> list[int] | None:
    if not pages:
        return None
    from .cli import parse_pages

    try:
        return parse_pages(pages)
    except Exception:
        raise HTTPException(status_code=422, detail=f"invalid pages: {pages!r} (use e.g. 1-3,5)") from None


def _read_upload(file: UploadFile) -> bytes:
    limit = int(MAX_UPLOAD_MB * 1024 * 1024)
    chunks, size = [], 0
    while chunk := file.file.read(1024 * 1024):
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail=f"file exceeds {MAX_UPLOAD_MB:g} MB")
        chunks.append(chunk)
    if not size:
        raise HTTPException(status_code=422, detail="empty file")
    return b"".join(chunks)


# ------------------------------------------------------------- endpoints
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/v1/extract", dependencies=[Depends(_check_key)])
def extract(
    file: UploadFile = File(..., description="PDF or image (PNG, JPEG, TIFF, BMP, WebP, GIF, JPEG 2000)"),
    content: Literal["all", "tables"] = Query("all", description="'all': full content; 'tables': tables only"),
    accuracy: Literal["fast", "balanced", "best"] = Query("balanced", description="OCR accuracy (content=all)"),
    output: Literal["json", "markdown", "text"] = Query("json"),
    pages: str | None = Query(None, description="1-based pages, e.g. 1-3,5"),
    password: str | None = Form(None, description="password for encrypted PDFs (form field, never in the URL)"),
):
    data = _read_upload(file)
    page_list = _parse_pages(pages)
    if not _slots.acquire(timeout=JOB_WAIT):
        raise HTTPException(status_code=503, detail="server busy, retry later", headers={"Retry-After": "30"})
    try:
        if content == "all":
            cfg = ImageDataConfig(
                accuracy=accuracy, password=password, max_file_size_mb=MAX_UPLOAD_MB, max_pages=MAX_PAGES, max_pixels=MAX_PIXELS
            )
            result = DocumentDataExtractor(cfg, models=_models(accuracy)).extract(data, pages=page_list)
            result.source = _safe_name(file.filename)
            if output == "markdown":
                return PlainTextResponse(result.to_markdown(), media_type="text/markdown")
            if output == "text":
                return PlainTextResponse(result.text)
            return JSONResponse(result.to_dict())

        cfg = ExtractionConfig(password=password, max_file_size_mb=MAX_UPLOAD_MB, max_pages=MAX_PAGES)
        pdf_engine, image_engine = _table_engines()
        if detect_type(data[:1024]) == "pdf":
            result = TableExtractor(cfg, ocr_engine=pdf_engine).extract(data, pages=page_list)
        else:
            result = ImageTableExtractor(cfg, ocr_engine=image_engine, max_pixels=MAX_PIXELS).extract(data, frames=page_list)
        result.source = _safe_name(file.filename)
        if output != "json":
            from .exporters import to_markdown

            md = to_markdown(result)
            return PlainTextResponse(md, media_type="text/markdown" if output == "markdown" else "text/plain")
        return JSONResponse(result.to_dict())
    finally:
        _slots.release()


# ------------------------------------------------------------ error map
def _error(status: int, code: str):
    def handler(_request, exc: Exception):
        return JSONResponse(status_code=status, content={"error": code, "detail": str(exc)})

    return handler


app.add_exception_handler(UnsupportedFileError, _error(415, "unsupported_file_type"))
app.add_exception_handler(EncryptedPDFError, _error(422, "encrypted_pdf"))
app.add_exception_handler(PDFTooLargeError, _error(413, "too_large"))
app.add_exception_handler(OCRUnavailableError, _error(503, "ocr_unavailable"))
app.add_exception_handler(ExtractionError, _error(422, "invalid_input"))
