"""Table structure recognition for document images (ruled and borderless).

Built for real-world photos and scans: coloured or hand-drawn rules, forms
with stacked boxes, handwritten rows without separating lines, and printed
invoices whose tables only have horizontal rules.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from statistics import median

import cv2
import numpy as np

from .grid import binarize
from .ocr_models import OCRLine, OCRModels

log = logging.getLogger(__name__)


# --------------------------------------------------------------- text tests
_DIGIT = re.compile(r"\d")
_SHORT_CODE = re.compile(r"^[A-Za-z]{1,2}\d{1,2}$")
# Item numbers as written by hand or printed: "1)", "2>", "(3", "4.", "5", "S.1"
_ENUMERATOR = re.compile(r"^[\(\[<]?\s*[0-9lIsS]{1,3}\s*[\)\]>.:,-]?$")


def is_label(text: str, max_words: int = 5) -> bool:
    """Short, mostly alphabetic text such as a column header or field name."""
    t = text.strip()
    if not t or len(t) > 40 or len(t.split()) > max_words:
        return False
    if _SHORT_CODE.match(t):
        return True  # column codes such as 'Q1', 'FY24', 'H2'
    letters = sum(ch.isalpha() for ch in t)
    digits = sum(ch.isdigit() for ch in t)
    return letters >= 2 and digits <= 0.3 * (letters + digits)


def is_header_cell(text: str) -> bool:
    """Column header: 1-3 words, letters only (no digits), e.g. 'UNIT PRICE'."""
    t = text.strip()
    return is_label(t, max_words=3) and (not any(ch.isdigit() for ch in t) or bool(_SHORT_CODE.match(t)))


_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_WORD_DATE = re.compile(
    rf"^(?:{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{2,4}}|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH},?\s+\d{{2,4}})$",
    re.IGNORECASE,
)


def is_value(text: str) -> bool:
    """A 'pure' value: amount, date, number or ID (digits dominate, few words),
    or a date written with a month name ('January 25, 2016', '5 Mar 2024')."""
    t = text.strip()
    if _WORD_DATE.match(t):
        return True
    if not t or len(t.split()) > 3:
        return False
    digits = sum(ch.isdigit() for ch in t)
    alnum = sum(ch.isalnum() for ch in t)
    return digits >= 1 and digits >= 0.5 * alnum


# ------------------------------------------------------------- structures
@dataclass
class Grid:
    bbox: tuple[int, int, int, int]
    row_edges: list[int]
    col_edges: list[int]


@dataclass
class TableCells:
    """A recognised table before conversion to the public Table model."""

    bbox: tuple[float, float, float, float]
    cells: list[list[list[OCRLine]]]  # rows x cols x lines
    method: str
    has_header: bool = True
    line_ids: set[int] = field(default_factory=set)
    footer_kv: list[tuple[OCRLine, OCRLine]] = field(default_factory=list)
    released: list[OCRLine] = field(default_factory=list)  # inside the frame but not table data
    warnings: list[str] = field(default_factory=list)

    def texts(self) -> list[list[str]]:
        return [[cell_text(c) for c in row] for row in self.cells]


def cell_text(lines: list[OCRLine]) -> str:
    if not lines:
        return ""
    ordered = sorted(lines, key=lambda ln: ln.cy)
    rows: list[list[OCRLine]] = [[ordered[0]]]
    for ln in ordered[1:]:
        ref = rows[-1][-1]
        if abs(ln.cy - ref.cy) <= 0.5 * max(min(ln.height, ref.height), 1):
            rows[-1].append(ln)
        else:
            rows.append([ln])
    return " ".join(" ".join(ln.text for ln in sorted(r, key=lambda ln: ln.bbox[0])) for r in rows).strip()


def _cluster(values: list[float], tol: float) -> list[int]:
    if not values:
        return []
    values = sorted(values)
    groups = [[values[0]]]
    for v in values[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(round(sum(g) / len(g))) for g in groups]


# ----------------------------------------------------------- ruled tables
def _segments(mask: np.ndarray, horizontal: bool, min_len: float, max_thick: float) -> list[tuple[int, int, int, int]]:
    """Thin, long connected components as (x0, y0, x1, y1)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        length, thick = (w, h) if horizontal else (h, w)
        if length < min_len:
            continue
        # Mean thickness (area / length) rejects filled shapes and text blobs.
        if area / max(length, 1) > max_thick:
            continue
        segs.append((int(x), int(y), int(x + w), int(y + h)))
    return segs


def find_ruled_grids(img: np.ndarray, char_h: float) -> list[Grid]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ink = binarize(gray)
    ch = max(char_h, 6.0)
    gap = max(3, int(0.3 * ch))
    h_src = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1)))
    v_src = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap)))
    h_len, v_len = int(max(4 * ch, 40)), int(max(2.2 * ch, 20))
    horiz = cv2.morphologyEx(h_src, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1)))
    vert = cv2.morphologyEx(v_src, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)))
    max_thick = max(4.0, 0.45 * ch)
    hs = _segments(horiz, True, h_len, max_thick)
    vs = _segments(vert, False, v_len, max_thick)
    if len(hs) < 2 or len(vs) < 2:
        return []

    lattice = np.zeros_like(ink)
    for x0, y0, x1, y1 in hs + vs:
        cv2.rectangle(lattice, (x0, y0), (x1 - 1, y1 - 1), 255, -1)
    lattice = cv2.dilate(lattice, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(lattice, connectivity=8)

    tol = max(4, int(0.4 * ch))
    grids: list[Grid] = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w < 6 * ch or h < 1.5 * ch:
            continue
        rh = [s for s in hs if s[0] >= x - 3 and s[2] <= x + w + 3 and s[1] >= y - 3 and s[3] <= y + h + 3]
        rv = [s for s in vs if s[0] >= x - 3 and s[2] <= x + w + 3 and s[1] >= y - 3 and s[3] <= y + h + 3]
        if len(rh) < 2 or len(rv) < 2:
            continue
        if float((ink[y : y + h, x : x + w] > 0).mean()) > 0.4:
            continue  # filled shape (logo, photo, stamp), not a grid
        rows = _cluster([(s[1] + s[3]) / 2 for s in rh if (s[2] - s[0]) >= 0.25 * w], tol)
        cols = _cluster([(s[0] + s[2]) / 2 for s in rv], tol)
        if len(rows) < 2 or len(cols) < 2:
            continue
        grids.extend(_split_by_signature(rows, cols, rv, tol, ch))
    grids.sort(key=lambda g: (g.bbox[1], g.bbox[0]))
    return grids


def _split_by_signature(rows: list[int], cols: list[int], vsegs, tol: int, ch: float) -> list[Grid]:
    """Split stacked boxes (e.g. a form header above an items table) where the
    set of vertical rules changes. Similar signatures (a sub-header that adds
    one divider) stay together."""
    bands = []
    for top, bot in zip(rows, rows[1:]):
        if bot - top < 0.8 * ch:
            continue  # sliver between two nearly touching boxes
        need = 0.6 * (bot - top)
        sig = frozenset(
            c for c in cols
            if any(abs((s[0] + s[2]) / 2 - c) <= tol and min(s[3], bot) - max(s[1], top) >= need for s in vsegs)
        )
        bands.append((top, bot, sig))

    groups: list[list[tuple[int, int, frozenset]]] = []
    for band in bands:
        if groups:
            prev = groups[-1][-1]
            union = set().union(*(b[2] for b in groups[-1]))
            inter_a, inter_b = set(union), set(band[2])
            jacc = len(inter_a & inter_b) / max(len(inter_a | inter_b), 1)
            contiguous = abs(band[0] - prev[1]) <= tol
            if contiguous and jacc >= 0.6:
                groups[-1].append(band)
                continue
        groups.append([band])

    grids = []
    for g in groups:
        edges = sorted(set().union(*(b[2] for b in g)))
        edges = _cluster(edges, tol)
        if len(edges) < 3:  # need at least two columns
            continue
        row_edges = [g[0][0]] + [b[1] for b in g]
        grids.append(Grid(bbox=(edges[0], row_edges[0], edges[-1], row_edges[-1]), row_edges=row_edges, col_edges=edges))
    return grids


def _col_overlaps(line: OCRLine, edges: list[int]) -> list[float]:
    x0, _, x1, _ = line.bbox
    width = max(x1 - x0, 1.0)
    return [max(0.0, min(x1, b) - max(x0, a)) / width for a, b in zip(edges, edges[1:])]


def _split_poly(poly: np.ndarray, xa: float, xb: float) -> np.ndarray:
    """Part of a text polygon between x=xa and x=xb (interpolated on its edges)."""
    x0, x1 = poly[:, 0].min(), poly[:, 0].max()
    ta, tb = (max(xa, x0) - x0) / max(x1 - x0, 1), (min(xb, x1) - x0) / max(x1 - x0, 1)
    top = lambda t: poly[0] + (poly[1] - poly[0]) * t
    bot = lambda t: poly[3] + (poly[2] - poly[3]) * t
    return np.array([top(ta), top(tb), bot(tb), bot(ta)])


def fill_grid(
    grid: Grid, lines: list[OCRLine], img: np.ndarray, models: OCRModels, char_h: float
) -> TableCells:
    x0, y0, x1, y1 = grid.bbox
    inside = [ln for ln in lines if x0 <= ln.cx <= x1 and y0 <= ln.cy <= y1]
    ncols = len(grid.col_edges) - 1
    placed: list[tuple[OCRLine, int]] = []
    warnings: list[str] = []
    next_id = max((ln.id for ln in lines), default=0) + 1000

    for ln in inside:
        ov = _col_overlaps(ln, grid.col_edges)
        best = int(np.argmax(ov))
        if ov[best] >= 0.75 or sum(o >= 0.2 for o in ov) < 2 or ln.model == "native":
            placed.append((ln, best))
            continue
        # One detection crosses a column rule: re-read each part separately.
        for c, o in enumerate(ov):
            if o < 0.2:
                continue
            poly = _split_poly(ln.poly, grid.col_edges[c], grid.col_edges[c + 1])
            from .ocr_models import crop_polygon

            name = "medium" if models.use_medium else "small"
            cand = models.recognize_crop(crop_polygon(img, poly), name) or models.recognize_crop(crop_polygon(img, poly), "small")
            if cand:
                part = OCRLine(id=next_id, poly=poly, text=cand.text, score=cand.score, model=cand.model)
                next_id += 1
                placed.append((part, c))
        warnings.append(f"text '{ln.text}' crossed a column rule and was re-read per cell")

    # Rows: ruled bands, sub-divided when one band holds several item rows.
    rows: list[list[list[OCRLine]]] = []
    released: list[OCRLine] = []
    for top, bot in zip(grid.row_edges, grid.row_edges[1:]):
        band = [(ln, c) for ln, c in placed if top <= ln.cy < bot]
        subs, overflow = _split_band(band, ncols, top, char_h)
        released.extend(overflow)
        for sub in subs:
            row = [[] for _ in range(ncols)]
            for ln, c in sub:
                row[c].append(ln)
            rows.append(row)
    if released:
        warnings.append(f"{len(released)} fragment(s) below the last item (stamp/signature?) kept outside the table")
    released_ids = {ln.id for ln in released}
    return TableCells(
        bbox=tuple(float(v) for v in grid.bbox),
        cells=rows,
        method="image-ruled",
        line_ids={ln.id for ln in inside} - released_ids,
        released=released,
        warnings=warnings,
    )


def _entries(items: list[tuple[OCRLine, int]]) -> list[list[OCRLine]]:
    """Group a column's lines into vertically separate entries."""
    lines = sorted((ln for ln, _ in items), key=lambda ln: ln.bbox[1])
    out: list[list[OCRLine]] = []
    for ln in lines:
        if out and ln.bbox[1] < max(o.bbox[3] for o in out[-1]) - 0.3 * ln.height:
            out[-1].append(ln)
        else:
            out.append([ln])
    return out


def _split_band(
    band: list[tuple[OCRLine, int]], ncols: int, top: int, char_h: float
) -> tuple[list[list[tuple[OCRLine, int]]], list[OCRLine]]:
    """Split a tall ruled band into item rows using an anchor column.

    Returns (rows, overflow) where overflow holds lines far below the last
    item (stamps, signatures) that do not belong to any row."""
    if not band:
        return [band], []
    per_col = [[(ln, c) for ln, c in band if c == col] for col in range(ncols)]
    entries = [_entries(items) for items in per_col]
    if max(len(e) for e in entries) < 2:
        return [band], []

    def anchor_score(col: int) -> tuple[int, int, int]:
        texts = [cell_text(e) for e in entries[col]]
        enum = sum(1 for t in texts if _ENUMERATOR.match(t))
        numeric = sum(1 for t in texts if is_value(t))
        return (enum, numeric, -col)

    candidates = [c for c in range(ncols) if len(entries[c]) >= 2]
    anchor = max(candidates, key=anchor_score)
    if sum(anchor_score(anchor)[:2]) < 2:
        return [band], []  # e.g. a two-line header cell: not item rows
    anchors = entries[anchor]
    starts = [min(ln.bbox[1] for ln in e) for e in anchors]
    centers = [sum(ln.cy for ln in e) / len(e) for e in anchors]
    lh = median(ln.height for e in anchors for ln in e)
    pitch = median([b - a for a, b in zip(starts, starts[1:])]) if len(starts) > 1 else 2 * lh
    last_bottom = max(ln.bbox[3] for ln in anchors[-1])
    # Text lines (wrapped descriptions) belong to the item above them unless
    # they start right at the next item; numeric values written slightly
    # above or below their item snap to the nearest item number.
    bounds = [top] + [s - 0.25 * lh for s in starts[1:]]
    rows: list[list[tuple[OCRLine, int]]] = [[] for _ in bounds]
    overflow: list[OCRLine] = []
    for ln, c in band:
        if ln.cy > last_bottom + 1.2 * pitch:
            overflow.append(ln)  # far below the last item: stamps, signatures
        elif c != anchor and is_value(ln.text):
            rows[min(range(len(centers)), key=lambda i: abs(centers[i] - ln.cy))].append((ln, c))
        else:
            rows[max(i for i, b in enumerate(bounds) if ln.cy >= b or i == 0)].append((ln, c))
    return [r for r in rows if r], overflow


# -------------------------------------------------------- borderless tables
def group_rows(lines: list[OCRLine]) -> list[list[OCRLine]]:
    """Lines sharing a text row (vertical overlap), each sorted left to right."""
    def overlaps(a: OCRLine, b: OCRLine) -> bool:
        ov = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
        return ov >= 0.5 * min(a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1])

    rows: list[list[OCRLine]] = []
    for ln in sorted(lines, key=lambda ln: ln.cy):
        # Join a row only if the line overlaps every member: a tall element
        # (logo text, script font) must not chain two text rows together.
        for row in rows:
            if all(overlaps(ln, r) for r in row):
                row.append(ln)
                break
        else:
            rows.append([ln])
    rows = [sorted(r, key=lambda ln: ln.bbox[0]) for r in rows]
    rows.sort(key=lambda r: min(ln.bbox[1] for ln in r))
    return rows


def find_borderless_tables(lines: list[OCRLine], char_h: float) -> list[TableCells]:
    """Header-driven detection: a row of >=3 short labels followed by rows whose
    cells align with those header columns (typical invoice line items)."""
    rows = group_rows(lines)
    used: set[int] = set()
    tables: list[TableCells] = []
    i = 0
    while i < len(rows):
        header = rows[i]
        if len(header) < 3 or not all(is_header_cell(ln.text) for ln in header) or any(ln.id in used for ln in header):
            i += 1
            continue
        spans = [(ln.bbox[0], ln.bbox[2]) for ln in header]
        bounds = [-1e9] + [(spans[k][1] + spans[k + 1][0]) / 2 for k in range(len(spans) - 1)] + [1e9]
        lh = median(ln.height for ln in header)

        def col_of(ln: OCRLine, b=bounds) -> int:
            return max(k for k in range(len(b) - 1) if ln.cx >= b[k])

        # Pass 1: collect the rows that belong to the table (provisional columns).
        body_rows: list[list[OCRLine]] = []
        continuation: list[bool] = []
        prev_bottom = max(ln.bbox[3] for ln in header)
        j = i + 1
        while j < len(rows):
            row = rows[j]
            gap = min(ln.bbox[1] for ln in row) - prev_bottom
            if gap > (4.5 if not body_rows else 3.5) * lh:
                break
            # A line spanning several header columns is prose, not a row.
            if any(sum(1 for a, b in zip(spans, spans[1:]) if ln.bbox[0] < a[0] and ln.bbox[2] > b[1]) for ln in row):
                break
            cols = {col_of(ln) for ln in row}
            if len(cols) >= 2:
                body_rows.append(row)
                continuation.append(False)
            elif body_rows and len(row) == 1 and gap <= 1.2 * lh and not is_value(row[0].text):
                body_rows.append(row)
                continuation.append(True)
            else:
                break
            prev_bottom = max(ln.bbox[3] for ln in row)
            j += 1

        # Pass 2: real column boundaries are the whitespace channels in the data.
        data_lines = [ln for r in body_rows for ln in r]
        bounds = [-1e9] + [_channel(data_lines, spans[k][0], spans[k + 1][1], bounds[k + 1]) for k in range(len(spans) - 1)] + [1e9]
        body: list[list[list[OCRLine]]] = []
        for row, is_cont in zip(body_rows, continuation):
            parts = [p for ln in row for p in _split_at_bounds(ln, spans, bounds)]
            if is_cont and body:
                for p in parts:
                    body[-1][col_of(p, bounds)].append(p)
                continue
            cells = [[] for _ in header]
            for p in parts:
                cells[col_of(p, bounds)].append(p)
            body.append(cells)

        if not body:
            i += 1
            continue
        all_lines = list(header) + [ln for r in body for c in r for ln in c]
        table = TableCells(
            bbox=(
                min(ln.bbox[0] for ln in all_lines),
                min(ln.bbox[1] for ln in all_lines),
                max(ln.bbox[2] for ln in all_lines),
                max(ln.bbox[3] for ln in all_lines),
            ),
            cells=[[[ln] for ln in header]] + body,
            method="image-borderless",
            line_ids={ln.id for ln in all_lines},
            warnings=["borderless table: columns inferred from the header row"],
        )
        extract_footer_kv(table)
        tables.append(table)
        used |= table.line_ids
        i = j
    return tables


def split_line_at(line: OCRLine, x: float) -> tuple[OCRLine, OCRLine] | None:
    """Split a detection into two lines at the word gap nearest to x.

    Uses the recogniser's word boxes when available (exact positions);
    otherwise falls back to proportional character positions.
    """
    x0, _, x1, _ = line.bbox
    if not (x0 < x < x1):
        return None
    words = line.word_boxes()
    if len(words) >= 2:
        gaps = []
        for i in range(len(words) - 1):
            right_of_a = float(words[i][1][:, 0].max())
            left_of_b = float(words[i + 1][1][:, 0].min())
            mid = (right_of_a + left_of_b) / 2
            gaps.append(((left_of_b - right_of_a) - 0.5 * abs(mid - x), i, mid))
        _, i, xs = max(gaps)
        left = " ".join(w for w, _ in words[: i + 1])
        right = " ".join(w for w, _ in words[i + 1 :])
    else:
        text = line.text
        spaces = [i for i, ch in enumerate(text) if ch == " "]
        if not spaces:
            return None
        target = (x - x0) / (x1 - x0) * len(text)
        cut = min(spaces, key=lambda i: abs(i - target))
        left, right = text[:cut], text[cut + 1 :]
        xs = x0 + (x1 - x0) * cut / len(text)
    left, right = left.strip(), right.strip()
    if not left or not right:
        return None
    a = OCRLine(id=line.id, poly=_split_poly(line.poly, x0, xs), text=left, score=line.score, model=line.model)
    b = OCRLine(id=line.id, poly=_split_poly(line.poly, xs, x1), text=right, score=line.score, model=line.model)
    a.words = [w for w in line.word_boxes() if w[1][:, 0].max() <= xs + 1]
    b.words = [w for w in line.word_boxes() if w[1][:, 0].min() >= xs - 1]
    return a, b


def _split_at_bounds(line: OCRLine, spans, bounds) -> list[OCRLine]:
    """Split a line whose text starts in one header column and continues into
    the next (e.g. '3 Labor 3hrs' under QTY | DESCRIPTION)."""
    x0, _, x1, _ = line.bbox
    width = max(x1 - x0, 1.0)
    for k in range(len(spans) - 1):
        bound = bounds[k + 1]
        # the text reaches well into both sides of a column boundary
        if x0 + 0.15 * width < bound < x1 - 0.15 * width:
            parts = split_line_at(line, bound)
            if parts:
                return [parts[0]] + _split_at_bounds(parts[1], spans, bounds)
    return [line]


def _channel(lines: list[OCRLine], lo: float, hi: float, fallback: float) -> float:
    """x inside [lo, hi] crossed by the fewest data lines (widest such run)."""
    if not lines or hi <= lo:
        return fallback
    xs = np.arange(int(lo), int(hi) + 1)
    counts = np.zeros(len(xs), dtype=int)
    for ln in lines:
        a, b = ln.bbox[0], ln.bbox[2]
        counts += (xs >= a) & (xs <= b)
    m = counts.min()
    best_len, best_mid, run_start = 0, fallback, None
    for idx, c in enumerate(list(counts) + [m + 1]):
        if c == m and run_start is None:
            run_start = idx
        elif c != m and run_start is not None:
            if idx - run_start > best_len:
                best_len, best_mid = idx - run_start, float(xs[run_start] + (idx - 1 - run_start) / 2)
            run_start = None
    return best_mid


def extract_footer_kv(table: TableCells) -> None:
    """Move trailing summary rows ('Subtotal | 145.00', 'TOTAL | $154.06') out
    of the item rows into key-value pairs."""
    while len(table.cells) > 2:
        row = table.cells[-1]
        filled = [i for i, c in enumerate(row) if c]
        if len(filled) != 2 or filled[0] == 0:
            break
        label, value = cell_text(row[filled[0]]), cell_text(row[filled[1]])
        if not (is_label(label) and is_value(value)):
            break
        table.footer_kv.insert(0, (row[filled[0]][0] if len(row[filled[0]]) == 1 else _merge(row[filled[0]]), _merge(row[filled[1]])))
        table.cells.pop()


def _merge(lines: list[OCRLine]) -> OCRLine:
    if len(lines) == 1:
        return lines[0]
    x0 = min(ln.bbox[0] for ln in lines)
    y0 = min(ln.bbox[1] for ln in lines)
    x1 = max(ln.bbox[2] for ln in lines)
    y1 = max(ln.bbox[3] for ln in lines)
    return OCRLine(
        id=lines[0].id,
        poly=np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float),
        text=cell_text(lines),
        score=min(ln.score for ln in lines),
        model=lines[0].model,
    )


NOISE_SCORE = 0.75


def drop_noise_rows(table: TableCells) -> list[OCRLine]:
    """Remove rows made only of low-confidence fragments (stamps, signatures,
    scribbles). Returns the removed lines so they can be reported."""
    removed: list[OCRLine] = []
    kept = []
    for i, row in enumerate(table.cells):
        lines = [ln for c in row for ln in c]
        noisy = lines and all(ln.score < NOISE_SCORE or len(ln.text.strip()) <= 2 for ln in lines) and i > 0
        if noisy and all(ln.score < NOISE_SCORE for ln in lines if len(ln.text.strip()) > 2):
            removed.extend(lines)
        else:
            kept.append(row)
    table.cells = kept
    if removed:
        table.warnings.append(f"{len(removed)} low-confidence fragment(s) (stamp/signature?) excluded from rows")
    return removed


def accept_ruled(table: TableCells) -> bool:
    """A ruled grid is a data table only if it has a header plus data rows, or
    is clearly tabular. Form boxes and one-row strips are released to
    key-value/text analysis instead."""
    rows = [r for r in table.cells if any(r)]
    if len(rows) < 2:
        return False
    substantive = sum(1 for r in rows for c in r if len(cell_text(c)) >= 2)
    if substantive < 3:
        return False  # e.g. letters inside a logo
    if header_is_labels(table):
        return True
    table.has_header = False
    return len(rows) >= 3 and len(table.cells[0]) >= 3


def header_is_labels(table: TableCells) -> bool:
    if not table.cells:
        return False
    first = [cell_text(c) for c in table.cells[0]]
    filled = [t for t in first if t]
    return len(filled) >= 2 and sum(is_label(t) for t in filled) >= max(2, 0.6 * len(filled))
