"""Exception hierarchy. Callers can catch ``ExtractionError`` for any failure."""


class ExtractionError(Exception):
    """Base class for all extraction failures."""


class InvalidPDFError(ExtractionError):
    """The input is missing, unreadable, or not a PDF."""


class EncryptedPDFError(ExtractionError):
    """The PDF is password protected and no valid password was supplied."""


class PDFTooLargeError(ExtractionError):
    """The PDF exceeds the configured size or page limits."""


class OCRUnavailableError(ExtractionError):
    """OCR is required for a page but the OCR engine could not be loaded."""
