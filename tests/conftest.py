"""Shared fixtures: synthetic PDFs built with PyMuPDF so tests are hermetic."""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_PDF = ROOT / "data" / "package.pdf"
DATA = ROOT / "data"


def draw_table(page: pymupdf.Page, x0: float, y0: float, col_w: list[float], rows: list[list[str]], row_h: float = 18) -> pymupdf.Rect:
    """Draw a fully ruled table with text and return its rectangle."""
    x_edges = [x0]
    for w in col_w:
        x_edges.append(x_edges[-1] + w)
    y_edges = [y0 + i * row_h for i in range(len(rows) + 1)]
    for y in y_edges:
        page.draw_line((x_edges[0], y), (x_edges[-1], y), width=0.6)
    for x in x_edges:
        page.draw_line((x, y_edges[0]), (x, y_edges[-1]), width=0.6)
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            page.insert_text((x_edges[c] + 4, y_edges[r] + row_h - 5), text, fontsize=9)
    return pymupdf.Rect(x_edges[0], y_edges[0], x_edges[-1], y_edges[-1])


def make_pdf(path: Path, pages: list[callable]) -> Path:
    doc = pymupdf.open()
    for build in pages:
        page = doc.new_page(width=595, height=842)
        build(page)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def simple_pdf(tmp_path: Path) -> Path:
    def page1(p):
        p.insert_text((40, 60), "Quarterly Revenue", fontsize=14)
        draw_table(p, 40, 80, [120, 80, 80], [["Region", "Q1", "Q2"], ["North", "1,200", "1,350"], ["South", "980", "1,010"]])
        p.insert_text((40, 800), "1", fontsize=8)  # page number, must not become a title

    return make_pdf(tmp_path / "simple.pdf", [page1])


@pytest.fixture
def continued_pdf(tmp_path: Path) -> Path:
    header = ["Item", "Qty", "Price"]

    def page1(p):
        p.insert_text((40, 60), "Parts List", fontsize=14)
        draw_table(p, 40, 80, [150, 60, 80], [header] + [[f"Part {i}", str(i), f"{i}.00"] for i in range(1, 6)])

    def page2(p):
        draw_table(p, 40, 40, [150, 60, 80], [header] + [[f"Part {i}", str(i), f"{i}.00"] for i in range(6, 9)])

    return make_pdf(tmp_path / "continued.pdf", [page1, page2])


@pytest.fixture
def sample_pdf() -> Path:
    if not SAMPLE_PDF.exists():
        pytest.skip("package.pdf sample not present")
    return SAMPLE_PDF
