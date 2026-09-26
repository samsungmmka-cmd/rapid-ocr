"""Page-level geometry types shared by extractors and title resolution."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import BBox


@dataclass
class TextLine:
    bbox: BBox
    text: str


@dataclass
class RawTable:
    """A table found on one page, before titles, merging and header split."""

    page: int
    bbox: BBox
    grid: list[list[str]]
    method: str
    confidence: float = 1.0
    low_confidence_cells: list[tuple[int, int, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    title: str | None = None


@dataclass
class PageData:
    page: int
    width: float
    height: float
    tables: list[RawTable]
    lines: list[TextLine]  # text lines outside any table, top to bottom
    warnings: list[str] = field(default_factory=list)


def contains(outer: BBox, inner: BBox, tol: float = 1.0) -> bool:
    return (
        inner[0] >= outer[0] - tol
        and inner[1] >= outer[1] - tol
        and inner[2] <= outer[2] + tol
        and inner[3] <= outer[3] + tol
    )


def center_inside(outer: BBox, inner: BBox) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]
