"""Writers for JSON, CSV and Markdown. All file writes are atomic."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import tempfile
from pathlib import Path

from .models import DocumentResult, Table

FORMATS = ("json", "csv", "md")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def slugify(text: str | None, max_len: int = 50) -> str:
    """Filesystem-safe slug (no separators, no traversal)."""
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:max_len].rstrip("-")


def to_json(result: DocumentResult, indent: int | None = 2) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=indent)


def table_to_csv(table: Table) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(table.column_keys())
    for row in table.rows:
        writer.writerow(row + [""] * (table.n_cols - len(row)))
    return buf.getvalue()


def _md_cell(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def table_to_markdown(table: Table) -> str:
    keys = table.column_keys()
    lines = [
        "| " + " | ".join(_md_cell(k) for k in keys) + " |",
        "|" + "---|" * len(keys),
    ]
    for row in table.rows:
        cells = row + [""] * (len(keys) - len(row))
        lines.append("| " + " | ".join(_md_cell(c) for c in cells) + " |")
    return "\n".join(lines)


def to_markdown(result: DocumentResult) -> str:
    out = [f"# Tables extracted from {Path(result.source).name}", ""]
    out.append(f"{len(result.tables)} table(s) from {result.n_pages} page(s); status: {'ok' if result.ok else 'partial'}.")
    if result.failed_pages:
        out.append(f"Failed pages: {', '.join(map(str, result.failed_pages))}.")
    notes = result.quality_notes() if hasattr(result, "quality_notes") else []
    if hasattr(result, "quality_level"):
        out += ["", f"## Source image quality: {result.quality_level}", ""]
        out += [f"- {n}" for n in notes] or ["- No quality problems detected."]
    for t in result.tables:
        pages = ", ".join(map(str, t.pages))
        out += ["", f"## Table {t.index + 1}: {t.title or 'Untitled'}", ""]
        out.append(f"Page {pages} · {t.method} · confidence {t.confidence:.2f} · {t.n_rows} rows × {t.n_cols} columns")
        out += ["", table_to_markdown(t)]
        for w in t.warnings:
            out.append(f"\n> Note: {w}")
        if t.low_confidence_cells:
            out.append(f"\n> Review: {len(t.low_confidence_cells)} low-confidence OCR cell(s).")
    return "\n".join(out) + "\n"


def write_outputs(result: DocumentResult, out_dir: Path, formats: tuple[str, ...] = FORMATS) -> list[Path]:
    """Write the requested formats under ``out_dir``; return written paths."""
    stem = slugify(Path(result.source).stem) or "document"
    written: list[Path] = []
    if "json" in formats:
        path = out_dir / f"{stem}.tables.json"
        _atomic_write(path, to_json(result) + "\n")
        written.append(path)
    if "csv" in formats:
        for t in result.tables:
            name = f"{stem}_table{t.index + 1:02d}"
            if slug := slugify(t.title):
                name += f"_{slug}"
            path = out_dir / f"{name}.csv"
            _atomic_write(path, table_to_csv(t))
            written.append(path)
    if "md" in formats:
        path = out_dir / f"{stem}.tables.md"
        _atomic_write(path, to_markdown(result))
        written.append(path)
    return written
