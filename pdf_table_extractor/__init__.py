"""Enterprise-grade extraction of structured tables from PDFs and images.

Pages with a text layer are parsed natively (exact text, no OCR errors).
Scanned or image-only pages fall back to RapidOCR plus ruled-grid detection.
Image files (PNG, JPEG, TIFF, BMP, WebP, GIF, JPEG 2000) are quality-checked,
enhanced when unclear, and carry quality notes in the result.

``ImageDataExtractor`` extracts *all* data from images (text in reading
order, headings, key-value fields, tables, uncertain fragments) using a
RapidOCR model ensemble for higher accuracy on hard text such as handwriting.
"""

from .config import ExtractionConfig, ExtractionMode
from .errors import (
    EncryptedPDFError,
    ExtractionError,
    InvalidPDFError,
    OCRUnavailableError,
    PDFTooLargeError,
)
from .models import DocumentResult, PageResult, Table
from .pipeline import TableExtractor, extract_tables
from .image_reader import ImageDocumentResult, ImageTableExtractor, InvalidImageError
from .reader import DocumentReader, UnsupportedFileError, extract_document
from .image_data import ImageDataConfig, ImageDataExtractor, ImageDataResult
from .document_data import DocumentDataExtractor, extract_content

__all__ = [
    "DocumentDataExtractor",
    "extract_content",
    "ImageDataConfig",
    "ImageDataExtractor",
    "ImageDataResult",
    "DocumentReader",
    "DocumentResult",
    "ImageDocumentResult",
    "ImageTableExtractor",
    "InvalidImageError",
    "UnsupportedFileError",
    "extract_document",
    "EncryptedPDFError",
    "ExtractionConfig",
    "ExtractionError",
    "ExtractionMode",
    "InvalidPDFError",
    "OCRUnavailableError",
    "PDFTooLargeError",
    "PageResult",
    "Table",
    "TableExtractor",
    "extract_tables",
]

__version__ = "1.3.0"
