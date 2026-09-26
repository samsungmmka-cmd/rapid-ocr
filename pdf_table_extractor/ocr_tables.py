"""Table extraction for scanned / image-only pages via RapidOCR."""

from __future__ import annotations

import logging
from statistics import median

import cv2
import numpy as np
import pymupdf

from .config import ExtractionConfig
from .grid import GridRegion, binarize, deskew, detect_ruled_tables, estimate_skew, ink_ratio
from .layout import PageData, RawTable, TextLine
from .normalize import clean_grid_indexed, clean_text, fix_ocr_text
from .ocr_engine import OCRBox, OCREngine

log = logging.getLogger(__name__)

MAX_PIXELS = 40_000_000  # hard cap on rendered page size (memory guard)
CELL_PAD = 3  # px trimmed inside each cell to avoid ruling lines
CROP_MARGIN = 12  # px of white added around a cell crop before re-OCR


def render_page(page: pymupdf.Page, dpi: int) -> tuple[np.ndarray, float]:
    """Render to a BGR array. Returns (image, pixels-per-point scale)."""
    scale = dpi / 72.0
    px = page.rect.width * page.rect.height * scale * scale
    if px > MAX_PIXELS:
        scale *= (MAX_PIXELS / px) ** 0.5
        log.warning("page %d: render scaled down to %.0f dpi (size cap)", page.number + 1, scale * 72)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
    rgb = arr[:, : pix.width * 3].reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), scale


def _join_boxes(boxes: list[OCRBox]) -> str:
    """Join boxes in reading order: group into lines by y, then left to right."""
    if not boxes:
        return ""
    boxes = sorted(boxes, key=lambda b: b.cy)
    line_tol = 0.5 * median(b.height for b in boxes)
    lines: list[list[OCRBox]] = [[boxes[0]]]
    for b in boxes[1:]:
        if abs(b.cy - lines[-1][-1].cy) <= line_tol:
            lines[-1].append(b)
        else:
            lines.append([b])
    return " ".join(" ".join(b.text for b in sorted(line, key=lambda b: b.bbox[0])) for line in lines)


def _index(edges: list[int], v: float) -> int:
    """Index of the interval [edges[i], edges[i+1]) containing v, or -1."""
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return i
    return -1


def _crop_ocr(engine: OCREngine, img: np.ndarray, box: tuple[int, int, int, int]) -> list[OCRBox]:
    x0, y0, x1, y1 = box
    crop = img[max(y0, 0) : y1, max(x0, 0) : x1]
    if crop.size == 0:
        return []
    crop = cv2.copyMakeBorder(crop, CROP_MARGIN, CROP_MARGIN, CROP_MARGIN, CROP_MARGIN, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    found = engine.recognize(crop, use_det=True)
    if not found:  # detector can miss tiny glyphs such as "5" or a dash
        found = engine.recognize(crop, use_det=False)
    return found


def _fill_region(
    region: GridRegion,
    boxes: list[OCRBox],
    img: np.ndarray,
    ink: np.ndarray,
    engine: OCREngine,
) -> tuple[list[list[str]], list[list[float | None]], list[str]]:
    rows, cols = region.n_rows, region.n_cols
    cell_boxes: list[list[list[OCRBox]]] = [[[] for _ in range(cols)] for _ in range(rows)]
    redo: set[tuple[int, int]] = set()
    warnings: list[str] = []

    for b in boxes:
        r = _index(region.row_edges, b.cy)
        c0 = _index(region.col_edges, b.bbox[0] + CELL_PAD)
        c1 = _index(region.col_edges, b.bbox[2] - CELL_PAD)
        if r < 0 or c0 < 0 or c1 < 0:
            continue
        if c0 != c1:  # one detection spans a column rule: re-read each cell alone
            redo.update((r, c) for c in range(c0, c1 + 1))
        else:
            cell_boxes[r][c0].append(b)

    def cell_rect(r: int, c: int) -> tuple[int, int, int, int]:
        return (
            region.col_edges[c] + CELL_PAD,
            region.row_edges[r] + CELL_PAD,
            region.col_edges[c + 1] - CELL_PAD,
            region.row_edges[r + 1] - CELL_PAD,
        )

    # Empty cells that still contain ink were likely missed by the detector.
    for r in range(rows):
        for c in range(cols):
            if not cell_boxes[r][c] and (r, c) not in redo and ink_ratio(ink, cell_rect(r, c)) > 0.004:
                redo.add((r, c))

    for r, c in sorted(redo):
        found = _crop_ocr(engine, img, cell_rect(r, c))
        if found:
            cell_boxes[r][c] = found
        elif ink_ratio(ink, cell_rect(r, c)) > 0.004:
            warnings.append(f"cell ({r},{c}) has ink but no text was recognised")

    texts = [[fix_ocr_text(clean_text(_join_boxes(cell_boxes[r][c]))) for c in range(cols)] for r in range(rows)]
    scores = [[min((b.score for b in cell_boxes[r][c]), default=None) for c in range(cols)] for r in range(rows)]
    return texts, scores, warnings


def _group_lines(boxes: list[OCRBox]) -> list[list[OCRBox]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b.cy)
    lines: list[list[OCRBox]] = [[boxes[0]]]
    for b in boxes[1:]:
        ref = lines[-1][-1]
        if abs(b.cy - ref.cy) <= 0.5 * max(min(b.height, ref.height), 1):
            lines[-1].append(b)
        else:
            lines.append([b])
    return [sorted(line, key=lambda b: b.bbox[0]) for line in lines]


def _borderless_tables(boxes: list[OCRBox], cfg: ExtractionConfig) -> list[tuple[tuple[float, float, float, float], list[list[OCRBox]], list[tuple[float, float]]]]:
    """Find whitespace-aligned tables: runs of multi-cell lines sharing columns."""
    lines = _group_lines(boxes)
    if not lines:
        return []
    line_h = median(b.height for b in boxes)
    runs: list[list[list[OCRBox]]] = []
    current: list[list[OCRBox]] = []
    for line in lines:
        gap_ok = not current or (line[0].cy - current[-1][0].cy) <= 3.0 * line_h
        if len(line) >= cfg.min_cols and gap_ok:
            current.append(line)
        else:
            if len(current) >= cfg.min_rows:
                runs.append(current)
            current = [line] if len(line) >= cfg.min_cols else []
    if len(current) >= cfg.min_rows:
        runs.append(current)

    tables = []
    for run in runs:
        intervals = sorted((b.bbox[0], b.bbox[2]) for line in run for b in line)
        bands: list[list[float]] = [list(intervals[0])]
        for x0, x1 in intervals[1:]:
            if x0 <= bands[-1][1] + 0.3 * line_h:
                bands[-1][1] = max(bands[-1][1], x1)
            else:
                bands.append([x0, x1])
        if len(bands) < cfg.min_cols:
            continue
        # Require most lines to populate most columns, otherwise it is prose.
        filled = [len({_band_of(bands, b.cx) for b in line}) for line in run]
        if median(filled) < max(cfg.min_cols, 0.6 * len(bands)):
            continue
        x0 = min(b.bbox[0] for line in run for b in line)
        y0 = min(b.bbox[1] for line in run for b in line)
        x1 = max(b.bbox[2] for line in run for b in line)
        y1 = max(b.bbox[3] for line in run for b in line)
        tables.append(((x0, y0, x1, y1), run, [tuple(bd) for bd in bands]))
    return tables


def _band_of(bands, x: float) -> int:
    for i, (b0, b1) in enumerate(bands):
        if b0 <= x <= b1:
            return i
    return min(range(len(bands)), key=lambda i: min(abs(x - bands[i][0]), abs(x - bands[i][1])))


def _low_cells(
    scores: list[list[float | None]], row_keep: list[int], col_keep: list[int], threshold: float
) -> list[tuple[int, int, float]]:
    """Cells under the review threshold, in coordinates of the cleaned grid."""
    out = []
    for r_new, r in enumerate(row_keep):
        for c_new, c in enumerate(col_keep):
            s = scores[r][c] if c < len(scores[r]) else None
            if s is not None and s < threshold:
                out.append((r_new, c_new, s))
    return out


def _table_confidence(scores: list[list[float | None]]) -> float:
    vals = [s for row in scores for s in row if s is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0


def extract_ocr(page: pymupdf.Page, page_no: int, cfg: ExtractionConfig, engine: OCREngine) -> PageData:
    img, scale = render_page(page, cfg.ocr_dpi)
    page_warnings: list[str] = []
    ink = binarize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if cfg.deskew:
        angle = estimate_skew(ink)
        if abs(angle) >= 0.05:
            log.info("page %d: deskewing by %.2f degrees", page_no, angle)
            img = deskew(img, angle)
            ink = binarize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            page_warnings.append(f"page deskewed by {angle:.2f} degrees")
    boxes = engine.recognize(img, use_det=True)
    log.debug("page %d: %d OCR boxes", page_no, len(boxes))

    def to_pt(b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return tuple(v / scale for v in b)  # type: ignore[return-value]

    tables: list[RawTable] = []
    used: set[int] = set()

    for region in detect_ruled_tables(img, scale, min_rows=1, min_cols=cfg.min_cols):
        rx0, ry0, rx1, ry1 = region.bbox
        inside = [i for i, b in enumerate(boxes) if rx0 <= b.cx <= rx1 and ry0 <= b.cy <= ry1]
        texts, scores, warns = _fill_region(region, [boxes[i] for i in inside], img, ink, engine)
        used.update(inside)
        grid, row_keep, col_keep = clean_grid_indexed(texts)
        if len(grid) < cfg.min_rows or not grid or len(grid[0]) < cfg.min_cols:
            continue
        low = _low_cells(scores, row_keep, col_keep, cfg.review_score_threshold)
        tables.append(
            RawTable(
                page=page_no,
                bbox=to_pt(region.bbox),
                grid=grid,
                method="ocr-ruled",
                confidence=_table_confidence(scores),
                low_confidence_cells=low,
                warnings=warns,
            )
        )

    if cfg.detect_borderless:
        rest = [b for i, b in enumerate(boxes) if i not in used]
        for bbox, run, bands in _borderless_tables(rest, cfg):
            cells: list[list[list[OCRBox]]] = [[[] for _ in bands] for _ in run]
            for r, line in enumerate(run):
                for b in line:
                    cells[r][_band_of(bands, b.cx)].append(b)
            texts = [[fix_ocr_text(clean_text(_join_boxes(c))) for c in row] for row in cells]
            scores = [[min((b.score for b in c), default=None) for c in row] for row in cells]
            grid, row_keep, col_keep = clean_grid_indexed(texts)
            if len(grid) < cfg.min_rows or len(grid[0]) < cfg.min_cols:
                continue
            ids = {id(b) for line in run for b in line}
            used.update(i for i, b in enumerate(boxes) if id(b) in ids)
            tables.append(
                RawTable(
                    page=page_no,
                    bbox=to_pt(bbox),
                    grid=grid,
                    method="ocr-borderless",
                    confidence=_table_confidence(scores) * 0.9,  # layout is inferred
                    low_confidence_cells=_low_cells(scores, row_keep, col_keep, cfg.review_score_threshold),
                    warnings=["borderless table: column layout inferred from text alignment"],
                )
            )

    tables.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
    remaining = [b for i, b in enumerate(boxes) if i not in used]
    lines = [
        TextLine(
            bbox=to_pt((min(b.bbox[0] for b in ln), min(b.bbox[1] for b in ln), max(b.bbox[2] for b in ln), max(b.bbox[3] for b in ln))),
            text=clean_text(" ".join(b.text for b in ln)),
        )
        for ln in _group_lines(remaining)
    ]
    return PageData(page=page_no, width=page.rect.width, height=page.rect.height, tables=tables, lines=lines, warnings=page_warnings)
