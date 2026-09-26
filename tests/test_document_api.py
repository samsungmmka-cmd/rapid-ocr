"""Full-content PDF extraction (DocumentDataExtractor) and the FastAPI service."""

import pymupdf
import pytest
from fastapi.testclient import TestClient

from pdf_table_extractor import DocumentDataExtractor, EncryptedPDFError, ImageDataConfig, extract_tables
from pdf_table_extractor import api as api_module

FAST = ImageDataConfig(accuracy="fast")


# --------------------------------------------------------- full content
def test_digital_pdf_full_content(sample_pdf):
    r = DocumentDataExtractor(FAST).extract(sample_pdf)
    assert [p.quality["method"] for p in r.pages] == ["native"] * 3
    # tables are exactly those of the unchanged table extractor
    assert [t.rows for t in r.tables] == [t.rows for t in extract_tables(sample_pdf).tables]
    assert r.tables[3].title == "4. Mobitel 1278 / 368 Packages"
    for text in ["SLT-MOBITEL Package List", "Official source pages", "• 5G Postpaid: www.mobitel.lk/sitestudio/5g/postpaid"]:
        assert text in r.text
    assert "Note" in {kv.key for kv in r.key_values}
    assert r.pages[0].blocks[0].type == "heading"


def test_boxed_invoice_pdf_uses_layout_analysis():
    path = pytest.importorskip("pathlib").Path(__file__).resolve().parent.parent / "data" / "invoice.pdf"
    if not path.exists():
        pytest.skip("data/invoice.pdf not present")
    r = DocumentDataExtractor(FAST).extract(path)
    fields = {kv.key: kv.value for kv in r.key_values}
    assert fields["Invoice Number"] == "INV-3337" and fields["Total"] == "$93.50" and fields["ACC #"] == "1234 1234"
    [t] = r.tables
    assert t.header == ["Hrs/Qty", "Service", "Rate/Price", "Adjust", "Sub Total"]


def test_form_fields_blank_and_scanned_pages(tmp_path, simple_pdf):
    doc = pymupdf.open(simple_pdf)
    page = doc[0]
    w = pymupdf.Widget()
    w.field_name, w.field_type, w.field_value = "customer_name", pymupdf.PDF_WIDGET_TYPE_TEXT, "Acme Ltd"
    w.rect = pymupdf.Rect(40, 300, 240, 320)
    page.add_widget(w)
    doc.new_page()  # blank page
    scan = doc.new_page(width=595, height=842)  # scanned copy of page 1
    scan.insert_image(scan.rect, pixmap=pymupdf.open(simple_pdf)[0].get_pixmap(dpi=200))
    path = tmp_path / "mixed.pdf"
    doc.save(path)

    r = DocumentDataExtractor(FAST).extract(path)
    assert [p.quality["method"] for p in r.pages] == ["native", "blank", "ocr"]
    assert ("customer_name", "Acme Ltd") in {(kv.key, kv.value) for kv in r.pages[0].key_values}
    assert r.pages[2].tables[0].rows == [["North", "1,200", "1,350"], ["South", "980", "1,010"]]


def test_garbled_text_layer_falls_back_to_ocr(monkeypatch, simple_pdf):
    import pdf_table_extractor.document_data as dd

    monkeypatch.setattr(dd, "_garbled", lambda page: True)
    r = DocumentDataExtractor(FAST).extract(simple_pdf)
    assert r.pages[0].quality["method"] == "ocr"
    assert any("garbled" in n for n in r.pages[0].quality["notes"])


def test_encrypted(tmp_path, simple_pdf):
    enc = tmp_path / "enc.pdf"
    pymupdf.open(simple_pdf).save(enc, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="o")
    with pytest.raises(EncryptedPDFError):
        DocumentDataExtractor(FAST).extract(enc)
    assert DocumentDataExtractor(ImageDataConfig(accuracy="fast", password="pw")).extract(enc).tables


# ------------------------------------------------------------------ API
@pytest.fixture
def client():
    return TestClient(api_module.app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_extract_pdf_all_and_tables(client, simple_pdf):
    files = {"file": ("simple.pdf", simple_pdf.read_bytes(), "application/pdf")}
    body = client.post("/v1/extract?accuracy=fast", files=files).json()
    assert body["source"] == "simple.pdf" and body["tables"][0]["header"] == ["Region", "Q1", "Q2"]
    assert "Quarterly Revenue" in body["text"]
    body = client.post("/v1/extract?content=tables", files=files).json()
    assert body["n_tables"] == 1 and body["tables"][0]["records"][0]["Region"] == "North"
    md = client.post("/v1/extract?accuracy=fast&output=markdown", files=files)
    assert md.headers["content-type"].startswith("text/markdown") and "| Region |" in md.text


@pytest.mark.ocr
def test_extract_image(client, simple_pdf):
    png = pymupdf.open(simple_pdf)[0].get_pixmap(dpi=200).tobytes("png")
    body = client.post("/v1/extract?accuracy=fast", files={"file": ("t.png", png, "image/png")}).json()
    assert body["input_type"] == "image" and body["tables"][0]["rows"][0] == ["North", "1,200", "1,350"]


def test_api_errors(client, simple_pdf, monkeypatch, tmp_path):
    bad = client.post("/v1/extract", files={"file": ("x.txt", b"hello", "text/plain")})
    assert bad.status_code == 415 and bad.json()["error"] == "unsupported_file_type"
    assert client.post("/v1/extract", files={"file": ("e.pdf", b"", "application/pdf")}).status_code == 422
    assert client.post("/v1/extract?pages=x", files={"file": ("s.pdf", simple_pdf.read_bytes())}).status_code == 422

    monkeypatch.setattr(api_module, "MAX_UPLOAD_MB", 0.001)
    assert client.post("/v1/extract", files={"file": ("s.pdf", simple_pdf.read_bytes())}).status_code == 413
    monkeypatch.setattr(api_module, "MAX_UPLOAD_MB", 50)

    enc = tmp_path / "enc.pdf"
    pymupdf.open(simple_pdf).save(enc, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="o")
    r = client.post("/v1/extract?accuracy=fast", files={"file": ("e.pdf", enc.read_bytes())})
    assert r.status_code == 422 and r.json()["error"] == "encrypted_pdf"
    r = client.post("/v1/extract?accuracy=fast", files={"file": ("e.pdf", enc.read_bytes())}, data={"password": "pw"})
    assert r.status_code == 200

    monkeypatch.setenv("API_KEY", "secret")
    assert client.post("/v1/extract", files={"file": ("s.pdf", simple_pdf.read_bytes())}).status_code == 401
    ok = client.post("/v1/extract?content=tables", files={"file": ("s.pdf", simple_pdf.read_bytes())}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
