"""Document-level assembly: titles, cross-page continuation, header split."""

from __future__ import annotations

import logging

from .config import ExtractionConfig
from .layout import PageData, RawTable, TextLine
from .models import Table
from .normalize import looks_like_page_number

log = logging.getLogger(__name__)


def _is_title_candidate(line: TextLine, cfg: ExtractionConfig) -> bool:
    return 0 < len(line.text) <= cfg.title_max_chars and not looks_like_page_number(line.text)


def _title_above(table: RawTable, page: PageData, cfg: ExtractionConfig) -> str | None:
    x0, y0, x1, _ = table.bbox
    best: TextLine | None = None
    for line in page.lines:
        lx0, _, _, ly1 = line.bbox
        if ly1 > y0 + 2 or y0 - ly1 > cfg.title_max_gap_pt or lx0 > x1:
            continue
        if not _is_title_candidate(line, cfg):
            continue
        # Reject if another table sits between this line and the table.
        if any(o is not table and o.bbox[1] >= ly1 - 1 and o.bbox[3] <= y0 + 1 for o in page.tables):
            continue
        if best is None or ly1 > best.bbox[3]:
            best = line
    return best.text if best else None


def _lines_above(table: RawTable, page: PageData) -> list[TextLine]:
    return [ln for ln in page.lines if ln.bbox[3] <= table.bbox[1] + 2 and not looks_like_page_number(ln.text)]


def _lines_below(table: RawTable | None, page: PageData) -> list[TextLine]:
    limit = table.bbox[3] - 2 if table else float("-inf")
    return [ln for ln in page.lines if ln.bbox[1] >= limit and not looks_like_page_number(ln.text)]


def _dangling_heading(page: PageData, cfg: ExtractionConfig) -> str | None:
    """A heading at the bottom of a page whose table starts on the next page."""
    last = page.tables[-1] if page.tables else None
    below = _lines_below(last, page)
    if not below:
        return None
    line = max(below, key=lambda ln: ln.bbox[3])
    return line.text if _is_title_candidate(line, cfg) and not line.text.endswith(".") else None


def _aligned(a: RawTable, b: RawTable, width: float) -> bool:
    tol = 0.02 * width
    return abs(a.bbox[0] - b.bbox[0]) <= tol and abs(a.bbox[2] - b.bbox[2]) <= tol


def _assign_titles(pages: list[PageData], cfg: ExtractionConfig, context: dict[int, PageData]) -> None:
    by_no = {p.page: p for p in pages}
    for page in pages:
        for t in page.tables:
            t.title = _title_above(t, page, cfg)
        if not page.tables:
            continue
        first = page.tables[0]
        prev = by_no.get(page.page - 1) or context.get(page.page - 1)
        if prev is not None and first.title is None and not _lines_above(first, page):
            heading = _dangling_heading(prev, cfg)
            if heading:
                first.title = heading
                first.warnings.append(f"title taken from bottom of page {prev.page}")


def _merge_continuations(pages: list[PageData], cfg: ExtractionConfig) -> list[tuple[RawTable, list[RawTable]]]:
    """Return (head, continuations) groups in document order."""
    groups: list[tuple[RawTable, list[RawTable]]] = []
    prev_page: PageData | None = None
    for page in pages:
        for j, t in enumerate(page.tables):
            if (
                cfg.merge_continued_tables
                and j == 0
                and prev_page is not None
                and prev_page.page == page.page - 1
                and prev_page.tables
                and groups
                and t.title is None
                and not _lines_above(t, page)
                and not _lines_below(prev_page.tables[-1], prev_page)
            ):
                head, conts = groups[-1]
                tail = conts[-1] if conts else head
                same_cols = len(t.grid[0]) == len(head.grid[0])
                repeated_header = t.grid[0] == head.grid[0]
                if same_cols and (repeated_header or _aligned(tail, t, page.width)):
                    conts.append(t)
                    continue
            groups.append((t, []))
        prev_page = page
    return groups


def _combine_header(header_rows: list[list[str]]) -> list[str]:
    if not header_rows:
        return []
    out = []
    for col in zip(*header_rows):
        parts: list[str] = []
        for part in col:
            if part and part not in parts:
                parts.append(part)
        out.append(" / ".join(parts))
    return out


def assemble(pages: list[PageData], cfg: ExtractionConfig, context: dict[int, PageData] | None = None) -> list[Table]:
    """Build final tables. ``context`` holds unselected neighbour pages that
    are only consulted for headings, never emitted."""
    _assign_titles(pages, cfg, context or {})
    tables: list[Table] = []
    for head, conts in _merge_continuations(pages, cfg):
        grid = [row[:] for row in head.grid]
        low = [(r, c, s) for r, c, s in head.low_confidence_cells]
        warnings = list(head.warnings)
        confidence = head.confidence
        pages_used = [head.page]
        for cont in conts:
            skip = 1 if cont.grid[0] == head.grid[0] else 0
            offset = len(grid) - skip
            grid.extend(cont.grid[skip:])
            low.extend((r + offset, c, s) for r, c, s in cont.low_confidence_cells if r >= skip)
            warnings.extend(cont.warnings)
            confidence = min(confidence, cont.confidence)
            pages_used.append(cont.page)
            warnings.append(f"continued on page {cont.page}" + (" (repeated header removed)" if skip else ""))

        h = min(cfg.header_rows, len(grid))
        tables.append(
            Table(
                page=head.page,
                index=len(tables),
                bbox=head.bbox,
                method=head.method,
                header=_combine_header(grid[:h]),
                rows=grid[h:],
                title=head.title,
                confidence=confidence,
                pages=pages_used,
                low_confidence_cells=[
                    {"section": "header" if r < h else "body", "row": r if r < h else r - h, "col": c, "score": s}
                    for r, c, s in low
                ],
                warnings=warnings,
            )
        )
    return tables
