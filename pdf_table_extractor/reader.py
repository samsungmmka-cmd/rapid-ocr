"""Single entry point for any supported document: PDF or image.

The file type is detected from its content (magic bytes), not its extension,
so misnamed files are handled. PDFs go to the existing, unchanged
:class:`TableExtractor`; images go to :class:`ImageTableExtractor`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from .config import ExtractionConfig
from .errors import InvalidPDFError
from .image_reader import IMAGE_EXTENSIONS, ImageTableExtractor, InvalidImageError
from .models import DocumentResult
from .pipeline import TableExtractor

PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS

_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
    (b"II+\x00", "tiff"),  # BigTIFF
    (b"MM\x00+", "tiff"),
    (b"BM", "bmp"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"\x00\x00\x00\x0cjP  \r\n\x87\n", "jp2"),
    (b"\xff\x4f\xff\x51", "j2k"),
    (b"P1", "pnm"), (b"P2", "pnm"), (b"P3", "pnm"), (b"P4", "pnm"), (b"P5", "pnm"), (b"P6", "pnm"),
)


class UnsupportedFileError(InvalidPDFError):
    """Neither a PDF nor a supported image format."""


def detect_type(data: bytes) -> str:
    """Return 'pdf' or 'image' from the file content, or raise."""
    if b"%PDF-" in data[:1024]:
        return "pdf"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image"
    for magic, _ in _IMAGE_MAGIC:
        if data.startswith(magic):
            return "image"
    raise UnsupportedFileError(
        "unsupported file type; expected PDF or an image (PNG, JPEG, TIFF, BMP, WebP, GIF, JPEG 2000, PNM)"
    )


def iter_inputs(paths: Iterable[Path], recursive: bool = False) -> Iterator[Path]:
    """Expand directories into supported files (sorted); pass files through."""
    for path in paths:
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in sorted(path.glob(pattern)):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield child
        else:
            yield path


class DocumentReader:
    """Extract tables from PDFs and images with one call."""

    def __init__(self, config: ExtractionConfig | None = None, enhance_images: bool = True):
        self.config = config or ExtractionConfig()
        self.pdf = TableExtractor(self.config)
        self.images = ImageTableExtractor(self.config, enhance_images=enhance_images)

    def extract(self, source: str | Path | bytes, pages: Iterable[int] | None = None) -> DocumentResult:
        if isinstance(source, (bytes, bytearray)):
            head = bytes(source[:1024])
        else:
            path = Path(source)
            if not path.is_file():
                raise InvalidPDFError(f"file not found: {path}")
            with path.open("rb") as fh:
                head = fh.read(1024)
        kind = detect_type(head)
        if kind == "pdf":
            return self.pdf.extract(source, pages=pages)
        return self.images.extract(source, frames=list(pages) if pages else None)


def extract_document(
    source: str | Path | bytes, config: ExtractionConfig | None = None, pages: Iterable[int] | None = None
) -> DocumentResult:
    """One-shot helper for PDFs and images."""
    return DocumentReader(config).extract(source, pages=pages)


__all__ = [
    "DocumentReader",
    "InvalidImageError",
    "SUPPORTED_EXTENSIONS",
    "UnsupportedFileError",
    "detect_type",
    "extract_document",
    "iter_inputs",
]
