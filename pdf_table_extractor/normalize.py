"""Cell-text and grid cleanup shared by the native and OCR paths."""

from __future__ import annotations

import re

_TRANSLATE = str.maketrans(
    {
        " ": " ",  # no-break space
        " ": " ",
        " ": " ",
        "​": None,  # zero-width space
        "‌": None,
        "‍": None,
        "﻿": None,
        "­": None,  # soft hyphen
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
    }
)
_WS = re.compile(r"\s+")
_PAGE_NUMBER = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)


def clean_text(value: object) -> str:
    """Normalise a cell value: None -> "", unify whitespace, drop invisibles."""
    if value is None:
        return ""
    text = str(value).translate(_TRANSLATE)
    return _WS.sub(" ", text).strip()


# Glyphs the Chinese/English recogniser returns for a horizontal dash.
_DASH_LOOKALIKES = {"一", "ー", "－", "―", "—", "–"}


def fix_ocr_text(text: str) -> str:
    """Narrow, safe OCR corrections applied only to OCR-derived cells."""
    if text and all(ch in _DASH_LOOKALIKES for ch in text):
        return "—"
    return text


def looks_like_page_number(text: str) -> bool:
    return bool(_PAGE_NUMBER.match(text.strip()))


def clean_grid_indexed(grid: list[list[object]]) -> tuple[list[list[str]], list[int], list[int]]:
    """Clean cells, pad ragged rows, drop fully empty rows/columns.

    Returns the grid plus the original row and column indices that were kept,
    so per-cell metadata (e.g. OCR scores) can be remapped.
    """
    cleaned = [[clean_text(c) for c in row] for row in grid]
    row_keep = [i for i, r in enumerate(cleaned) if any(r)]
    if not row_keep:
        return [], [], []
    rows = [cleaned[i] for i in row_keep]
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    col_keep = [j for j in range(width) if any(r[j] for r in rows)]
    return [[r[j] for j in col_keep] for r in rows], row_keep, col_keep


def clean_grid(grid: list[list[object]]) -> list[list[str]]:
    return clean_grid_indexed(grid)[0]
