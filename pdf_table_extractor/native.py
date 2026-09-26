"""Table extraction from the PDF text layer (digital PDFs). Exact, no OCR."""

from __future__ import annotations

import logging

import pymupdf

from .config import ExtractionConfig
from .layout import PageData, RawTable, TextLine, center_inside
from .normalize import clean_grid, clean_text

log = logging.getLogger(__name__)

if hasattr(pymupdf, "no_recommend_layout"):
    pymupdf.no_recommend_layout()  # keep stdout clean for JSON output


def native_word_count(page: pymupdf.Page) -> int:
    return len(page.get_text("words"))


def image_coverage(page: pymupdf.Page) -> float:
    """Share of the page area covered by raster images (0..1)."""
    page_area = abs(page.rect)
    if not page_area:
        return 0.0
    covered = 0.0
    for info in page.get_image_info():
        rect = pymupdf.Rect(info["bbox"]) & page.rect
        covered += abs(rect)
    return min(covered / page_area, 1.0)


def _text_lines(page: pymupdf.Page, exclude: list[tuple[float, float, float, float]]) -> list[TextLine]:
    lines: list[TextLine] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = clean_text("".join(span["text"] for span in line["spans"]))
            bbox = tuple(line["bbox"])
            if text and not any(center_inside(t, bbox) for t in exclude):
                lines.append(TextLine(bbox=bbox, text=text))
    lines.sort(key=lambda ln: (round(ln.bbox[1]), ln.bbox[0]))
    return lines


def extract_native(page: pymupdf.Page, page_no: int, cfg: ExtractionConfig) -> PageData:
    finder = page.find_tables(strategy=cfg.native_strategy)
    tables: list[RawTable] = []
    for found in finder.tables:
        grid = clean_grid(found.extract())
        if len(grid) < cfg.min_rows or (grid and len(grid[0]) < cfg.min_cols):
            log.debug("page %d: skipping %dx%d native table", page_no, len(grid), len(grid[0]) if grid else 0)
            continue
        tables.append(RawTable(page=page_no, bbox=tuple(found.bbox), grid=grid, method="native"))

    tables.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
    return PageData(
        page=page_no,
        width=page.rect.width,
        height=page.rect.height,
        tables=tables,
        lines=_text_lines(page, [t.bbox for t in tables]),
    )
