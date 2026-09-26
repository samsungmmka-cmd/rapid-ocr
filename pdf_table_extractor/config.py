"""Runtime configuration for the extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ExtractionMode(str, Enum):
    """How each page is processed."""

    AUTO = "auto"  # native text layer when present, OCR otherwise
    NATIVE = "native"  # text layer only, never OCR
    OCR = "ocr"  # always OCR, ignore the text layer


@dataclass(frozen=True)
class ExtractionConfig:
    mode: ExtractionMode = ExtractionMode.AUTO

    # --- Safety limits -----------------------------------------------------
    max_file_size_mb: float = 200.0
    max_pages: int = 2000
    password: str | None = field(default=None, repr=False)

    # --- Page classification -----------------------------------------------
    # A page needs at least this many native words to count as "digital".
    min_native_words: int = 5
    # If images cover this share of a digital page and no native table was
    # found, the page is treated as scanned and OCR is attempted.
    image_coverage_ocr_threshold: float = 0.5

    # --- Native extraction ---------------------------------------------------
    # PyMuPDF strategy: "lines" (ruled tables, precise) or "text" (borderless).
    native_strategy: str = "lines"

    # --- OCR extraction ------------------------------------------------------
    ocr_dpi: int = 200
    ocr_min_score: float = 0.5
    # Cells whose OCR confidence is below this are flagged for human review.
    review_score_threshold: float = 0.85
    # Detect whitespace-aligned tables without ruling lines on OCR pages.
    detect_borderless: bool = False
    ocr_max_side_len: int = 4000
    # Straighten scanned pages before grid detection (+/- 5 degrees).
    deskew: bool = True

    # --- Table post-processing ---------------------------------------------
    header_rows: int = 1
    min_rows: int = 2
    min_cols: int = 2
    # Max distance (PDF points) between a heading and the table below it.
    title_max_gap_pt: float = 40.0
    title_max_chars: int = 150
    merge_continued_tables: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.mode, str) and not isinstance(self.mode, ExtractionMode):
            object.__setattr__(self, "mode", ExtractionMode(self.mode))
        if not 72 <= self.ocr_dpi <= 600:
            raise ValueError("ocr_dpi must be between 72 and 600")
        if not 0.0 <= self.ocr_min_score <= 1.0:
            raise ValueError("ocr_min_score must be between 0 and 1")
        if not 0.0 <= self.review_score_threshold <= 1.0:
            raise ValueError("review_score_threshold must be between 0 and 1")
        if self.header_rows < 0:
            raise ValueError("header_rows must be >= 0")
        if self.max_pages < 1 or self.max_file_size_mb <= 0:
            raise ValueError("max_pages and max_file_size_mb must be positive")
        if self.native_strategy not in {"lines", "lines_strict", "text"}:
            raise ValueError("native_strategy must be lines, lines_strict or text")
