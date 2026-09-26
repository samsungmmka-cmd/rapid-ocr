from pathlib import Path

import pymupdf
import pytest

from pdf_table_extractor import (
    EncryptedPDFError,
    ExtractionConfig,
    InvalidPDFError,
    PDFTooLargeError,
    TableExtractor,
    extract_tables,
)
from pdf_table_extractor.ocr_engine import OCREngine


class RecordingOCR(OCREngine):
    """Fake engine: proves routing without loading models."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def recognize(self, img_bgr, use_det=True):
        self.calls += 1
        return []


def rasterize(src: Path, dst: Path, dpi: int = 150) -> Path:
    """Turn a digital PDF into an image-only (scanned-like) PDF."""
    doc, out = pymupdf.open(src), pymupdf.open()
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, pixmap=pix)
    out.save(dst)
    return dst


def test_native_extraction_simple(simple_pdf):
    result = extract_tables(simple_pdf)
    assert result.ok
    [t] = result.tables
    assert t.method == "native"
    assert t.title == "Quarterly Revenue"
    assert t.header == ["Region", "Q1", "Q2"]
    assert t.records() == [
        {"Region": "North", "Q1": "1,200", "Q2": "1,350"},
        {"Region": "South", "Q1": "980", "Q2": "1,010"},
    ]
    assert len(result.sha256) == 64


def test_bytes_input(simple_pdf):
    result = extract_tables(simple_pdf.read_bytes())
    assert result.source == "<bytes>"
    assert len(result.tables) == 1


def test_table_continued_across_pages_is_merged(continued_pdf):
    [t] = extract_tables(continued_pdf).tables
    assert t.pages == [1, 2]
    assert [r[0] for r in t.rows] == [f"Part {i}" for i in range(1, 9)]
    assert t.title == "Parts List"


def test_invalid_inputs(tmp_path):
    with pytest.raises(InvalidPDFError):
        extract_tables(tmp_path / "missing.pdf")
    bogus = tmp_path / "bogus.pdf"
    bogus.write_text("not a pdf")
    with pytest.raises(InvalidPDFError):
        extract_tables(bogus)
    truncated = tmp_path / "truncated.pdf"
    truncated.write_bytes(b"%PDF-1.4\n garbage")
    with pytest.raises(InvalidPDFError):
        extract_tables(truncated)


def test_page_out_of_range(simple_pdf):
    with pytest.raises(InvalidPDFError):
        extract_tables(simple_pdf, pages=[2])


def test_limits(simple_pdf, continued_pdf):
    with pytest.raises(PDFTooLargeError):
        extract_tables(continued_pdf, ExtractionConfig(max_pages=1))
    with pytest.raises(PDFTooLargeError):
        extract_tables(simple_pdf, ExtractionConfig(max_file_size_mb=0.0001))


def test_encrypted_pdf(tmp_path, simple_pdf):
    enc = tmp_path / "enc.pdf"
    doc = pymupdf.open(simple_pdf)
    doc.save(enc, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="s3cret", owner_pw="owner")
    with pytest.raises(EncryptedPDFError):
        extract_tables(enc)
    with pytest.raises(EncryptedPDFError):
        extract_tables(enc, ExtractionConfig(password="wrong"))
    assert len(extract_tables(enc, ExtractionConfig(password="s3cret")).tables) == 1


def test_config_validation():
    with pytest.raises(ValueError):
        ExtractionConfig(ocr_dpi=10)
    with pytest.raises(ValueError):
        ExtractionConfig(mode="bogus")
    assert ExtractionConfig(mode="ocr").mode.value == "ocr"


def test_routing_blank_native_and_scanned(tmp_path, simple_pdf):
    blank = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(blank)
    fake = RecordingOCR()
    res = TableExtractor(ocr_engine=fake).extract(blank)
    assert res.pages[0].method == "skipped" and fake.calls == 0

    res = TableExtractor(ocr_engine=fake).extract(simple_pdf)
    assert res.pages[0].method == "native" and fake.calls == 0

    scanned = rasterize(simple_pdf, tmp_path / "scan.pdf")
    res = TableExtractor(ocr_engine=fake).extract(scanned)
    assert res.pages[0].method == "ocr" and fake.calls >= 1

    res = TableExtractor(ExtractionConfig(mode="native"), ocr_engine=fake).extract(scanned)
    assert res.pages[0].method == "native" and res.tables == []


def test_page_failure_is_isolated(continued_pdf, monkeypatch):
    import pdf_table_extractor.pipeline as pipeline

    real = pipeline.extract_native

    def flaky(page, page_no, cfg):
        if page_no == 2:
            raise RuntimeError("simulated corruption")
        return real(page, page_no, cfg)

    monkeypatch.setattr(pipeline, "extract_native", flaky)
    result = extract_tables(continued_pdf)
    assert result.failed_pages == [2]
    assert not result.ok
    assert len(result.tables) == 1 and result.tables[0].pages == [1]
    assert "simulated corruption" in result.pages[1].error


# ---------------------------------------------------------------- real OCR --
@pytest.mark.ocr
def test_real_ocr_on_scanned_table(tmp_path, simple_pdf):
    scanned = rasterize(simple_pdf, tmp_path / "scan.pdf", dpi=200)
    result = extract_tables(scanned)
    assert result.pages[0].method == "ocr"
    [t] = result.tables
    assert t.method == "ocr-ruled"
    assert t.header == ["Region", "Q1", "Q2"]
    assert t.rows == [["North", "1,200", "1,350"], ["South", "980", "1,010"]]
    assert t.title == "Quarterly Revenue"
    assert 0.8 <= t.confidence <= 1.0
