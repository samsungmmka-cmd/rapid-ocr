"""Result data model. Everything serialises to plain JSON-compatible dicts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF points


@dataclass
class Table:
    page: int  # 1-based page number where the table starts
    index: int  # 0-based position of the table in the document
    bbox: BBox
    method: str  # "native" | "ocr-ruled" | "ocr-borderless"
    header: list[str]
    rows: list[list[str]]
    title: str | None = None
    confidence: float = 1.0
    pages: list[int] = field(default_factory=list)
    # OCR cells below the review threshold:
    # {"section": "header"|"body", "row": int, "col": int, "score": float}
    low_confidence_cells: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.pages:
            self.pages = [self.page]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max([len(self.header)] + [len(r) for r in self.rows]) if (self.header or self.rows) else 0

    def column_keys(self) -> list[str]:
        """Unique, non-empty keys derived from the header for record output."""
        keys: list[str] = []
        seen: dict[str, int] = {}
        for i in range(self.n_cols):
            name = self.header[i].strip() if i < len(self.header) else ""
            name = name or f"column_{i + 1}"
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 1
            keys.append(name)
        return keys

    def records(self) -> list[dict[str, str]]:
        keys = self.column_keys()
        return [{k: (row[i] if i < len(row) else "") for i, k in enumerate(keys)} for row in self.rows]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "page": self.page,
            "pages": self.pages,
            "bbox": [round(v, 2) for v in self.bbox],
            "method": self.method,
            "confidence": round(self.confidence, 4),
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": self.column_keys(),
            "header": self.header,
            "rows": self.rows,
            "records": self.records(),
            "low_confidence_cells": [{**cell, "score": round(cell["score"], 4)} for cell in self.low_confidence_cells],
            "warnings": self.warnings,
        }


@dataclass
class PageResult:
    page: int
    method: str  # "native" | "ocr" | "skipped" | "error"
    n_tables: int = 0
    duration_ms: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "method": self.method,
            "n_tables": self.n_tables,
            "duration_ms": round(self.duration_ms, 1),
            "error": self.error,
            "warnings": self.warnings,
        }


@dataclass
class DocumentResult:
    source: str
    sha256: str
    n_pages: int
    mode: str
    tables: list[Table] = field(default_factory=list)
    pages: list[PageResult] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def failed_pages(self) -> list[int]:
        return [p.page for p in self.pages if p.error]

    @property
    def ok(self) -> bool:
        return not self.failed_pages

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source": self.source,
            "sha256": self.sha256,
            "n_pages": self.n_pages,
            "mode": self.mode,
            "n_tables": len(self.tables),
            "status": "ok" if self.ok else "partial",
            "failed_pages": self.failed_pages,
            "duration_ms": round(self.duration_ms, 1),
            "pages": [p.to_dict() for p in self.pages],
            "tables": [t.to_dict() for t in self.tables],
        }
