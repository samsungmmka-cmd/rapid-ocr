"""Ruled-table detection on rendered page images using morphology (OpenCV)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GridRegion:
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels
    row_edges: list[int]  # sorted y coordinates of horizontal rules
    col_edges: list[int]  # sorted x coordinates of vertical rules

    @property
    def n_rows(self) -> int:
        return max(len(self.row_edges) - 1, 0)

    @property
    def n_cols(self) -> int:
        return max(len(self.col_edges) - 1, 0)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink mask (255 = ink). Adaptive threshold copes with uneven scans."""
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 8)
    _, global_ = cv2.threshold(gray, 228, 255, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(adaptive, global_) if gray.mean() > 200 else adaptive


def _cluster(values: list[int], tol: int) -> list[int]:
    if not values:
        return []
    values = sorted(values)
    groups: list[list[int]] = [[values[0]]]
    for v in values[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(round(sum(g) / len(g))) for g in groups]


def _with_border(edges: list[int], length: int, tol: int) -> list[int]:
    """The lattice bounding box is the table's outer border: always an edge.

    Scanned outer rules are often broken, so they are added here even when
    their pixel coverage alone would not qualify.
    """
    edges = sorted(edges)
    margin = 3 * tol
    if not edges or edges[0] > margin:
        edges.insert(0, 2)  # component bbox includes ~2px of dilation
    if edges[-1] < length - 1 - margin:
        edges.append(length - 3)
    return edges


def _spanning(mask: np.ndarray, edges: list[int], cross: list[int], tol: int, axis: int) -> list[int]:
    """Filter ``edges`` to rules that cover >=90% of some interval of ``cross``.

    axis=0: edges are y positions of horizontal rules, ``cross`` are x positions.
    axis=1: edges are x positions of vertical rules, ``cross`` are y positions.
    """
    if len(cross) < 2:
        return edges
    kept = []
    for e in edges:
        lo, hi = max(e - tol, 0), e + tol + 1
        band = mask[lo:hi, :].max(axis=0) if axis == 0 else mask[:, lo:hi].max(axis=1)
        for a, b in zip(cross, cross[1:]):
            seg = band[a:b]
            if seg.size and (seg > 0).mean() >= 0.9:
                kept.append(e)
                break
    return kept


def detect_ruled_tables(
    img_bgr: np.ndarray, scale: float, min_rows: int = 1, min_cols: int = 2
) -> list[GridRegion]:
    """Find tables drawn with horizontal and vertical ruling lines.

    ``scale`` is pixels per PDF point, used to express thresholds in points so
    behaviour is independent of the render DPI.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    ink = binarize(gray)

    h_len = max(15, int(25 * scale))  # a rule is at least ~25pt long
    v_len = max(10, int(10 * scale))
    gap = max(3, int(1.5 * scale))  # close breaks from noise / JPEG before opening
    h_src = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1)))
    v_src = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap)))
    horiz = cv2.morphologyEx(h_src, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1)))
    vert = cv2.morphologyEx(v_src, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)))

    # Bridge small breaks in scanned lines.
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))

    lattice = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(lattice, connectivity=8)

    tol = max(2, int(2 * scale))
    regions: list[GridRegion] = []
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        if w < 40 * scale or h < 8 * scale:
            continue
        h_roi = horiz[y : y + h, x : x + w]
        v_roi = vert[y : y + h, x : x + w]

        # Candidate rules: rows / columns with enough line pixels.
        row_cov = (h_roi > 0).sum(axis=1)
        rows = _cluster([int(r) for r in np.where(row_cov >= 0.1 * w)[0]], tol)
        col_cov = (v_roi > 0).sum(axis=0)
        cols = _cluster([int(c) for c in np.where(col_cov >= min(0.1 * h, 8 * scale))[0]], tol)
        # Keep only rules that fully span at least one cell between two
        # perpendicular rules. Bridged text inside a padded cell never does,
        # while partial rules around merged cells still qualify.
        rows = _with_border(_spanning(h_roi, rows, cols, tol, axis=0), h, tol)
        cols = _with_border(_spanning(v_roi, cols, rows, tol, axis=1), w, tol)
        rows = _with_border(_spanning(h_roi, rows, cols, tol, axis=0), h, tol)
        row_edges = [y + r for r in rows]
        col_edges = [x + c for c in cols]

        region = GridRegion(bbox=(int(x), int(y), int(x + w), int(y + h)), row_edges=row_edges, col_edges=col_edges)
        if region.n_rows >= min_rows and region.n_cols >= min_cols:
            regions.append(region)

    regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
    return regions


def ink_ratio(ink: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = ink[y0:y1, x0:x1]
    return float((roi > 0).mean()) if roi.size else 0.0


def estimate_skew(ink: np.ndarray, max_angle: float = 5.0, step: float = 0.1) -> float:
    """Estimate page skew in degrees via horizontal projection-profile variance.

    Text lines and table rules produce the sharpest row profile when level.
    Positive result means the content is rotated counter-clockwise.
    """
    h, w = ink.shape[:2]
    factor = min(1.0, 1000.0 / max(h, w))
    small = cv2.resize(ink, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA) if factor < 1 else ink
    sh, sw = small.shape[:2]
    center = (sw / 2, sh / 2)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(small, m, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        profile = rotated.sum(axis=1, dtype=np.float64)
        return float(np.var(profile))

    coarse = np.arange(-max_angle, max_angle + 1e-9, 0.5)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.5, best + 0.5 + 1e-9, step)
    return float(max(fine, key=score))


def deskew(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
