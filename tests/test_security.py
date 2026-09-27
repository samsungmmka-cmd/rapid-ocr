"""Regression tests for the security review findings (resource exhaustion,
upload limits, untrusted filenames)."""

import io

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pdf_table_extractor import PDFTooLargeError
from pdf_table_extractor import api as api_module
from pdf_table_extractor.api import _safe_name
from pdf_table_extractor.image_data import render_pdf_page
from pdf_table_extractor.image_reader import load_frames


def _tiff(n_frames: int, size=(40, 30)) -> bytes:
    frames = [Image.new("RGB", size, "white") for _ in range(n_frames)]
    buf = io.BytesIO()
    frames[0].save(buf, "TIFF", save_all=True, append_images=frames[1:])
    return buf.getvalue()


def test_frames_are_decoded_lazily_and_validated_up_front():
    frames, fmt, _ = load_frames(_tiff(5), max_frames=10, max_pixels=10_000)
    assert len(frames) == 5 and fmt == "TIFF"
    assert frames[4].shape == (30, 40, 3)
    with pytest.raises(PDFTooLargeError):
        load_frames(_tiff(5), max_frames=4, max_pixels=10_000)  # frame-count bomb
    with pytest.raises(PDFTooLargeError):
        load_frames(_tiff(2, size=(200, 200)), max_frames=10, max_pixels=10_000)  # pixel bomb


def test_pillow_global_limits_are_not_modified():
    before = Image.MAX_IMAGE_PIXELS
    with pytest.raises(PDFTooLargeError):
        load_frames(_tiff(1), max_frames=10, max_pixels=50)
    assert Image.MAX_IMAGE_PIXELS == before


def test_pdf_render_is_pixel_capped():
    doc = pymupdf.open()
    page = doc.new_page(width=14400, height=14400)  # 200 x 200 inches, the PDF maximum
    img = render_pdf_page(page, dpi=300, max_pixels=4_000_000)
    assert img.shape[0] * img.shape[1] <= 4_000_000


def test_safe_filename():
    assert _safe_name("../../etc/passwd") == "passwd"
    assert _safe_name("C:\\\\evil\\\\x.pdf") == "x.pdf"
    assert _safe_name("a\nb\x00.png") == "ab.png"
    assert _safe_name(None) == "upload" and len(_safe_name("x" * 1000)) == 128


@pytest.fixture
def client():
    return TestClient(api_module.app)


def test_oversized_upload_rejected_before_parsing(client, monkeypatch):
    monkeypatch.setattr(api_module, "MAX_UPLOAD_MB", 0.01)  # ~10 KB (+64 KB multipart allowance)
    big = b"%PDF-" + b"0" * 200_000
    r = client.post("/v1/extract", files={"file": ("big.pdf", big)})
    assert r.status_code == 413 and r.json()["error"] == "too_large"

    def chunks():  # valid multipart, no Content-Length: streamed bytes are counted
        yield b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n\r\n%PDF-'
        for _ in range(20):
            yield b"0" * 10_000
        yield b"\r\n--x--\r\n"

    r = client.post("/v1/extract", content=chunks(), headers={"content-type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413


def test_page_limit(client, monkeypatch, simple_pdf):
    doc = pymupdf.open(simple_pdf)
    doc.insert_pdf(pymupdf.open(simple_pdf))
    monkeypatch.setattr(api_module, "MAX_PAGES", 1)
    r = client.post("/v1/extract?content=tables", files={"file": ("two.pdf", doc.tobytes())})
    assert r.status_code == 413
    r = client.post("/v1/extract?accuracy=fast", files={"file": ("two.pdf", doc.tobytes())})
    assert r.status_code == 413


def test_response_uses_sanitised_filename(client, simple_pdf):
    r = client.post("/v1/extract?content=tables", files={"file": ("../../secret/q.pdf", simple_pdf.read_bytes())})
    assert r.json()["source"] == "q.pdf"


def test_page_range_expansion_is_bounded(client, simple_pdf):
    from pdf_table_extractor.cli import parse_pages

    with pytest.raises(Exception):
        parse_pages("1-100000000")
    r = client.post("/v1/extract?pages=1-100000000", files={"file": ("s.pdf", simple_pdf.read_bytes())})
    assert r.status_code == 422


def test_production_requires_strong_key(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_KEY_FILE", raising=False)
    with pytest.raises(RuntimeError):
        with TestClient(api_module.app):  # startup runs the check
            pass
    monkeypatch.setenv("API_KEY", "short")
    with pytest.raises(RuntimeError):
        api_module.validate_security_config()


def test_key_file_and_rotation(monkeypatch, tmp_path, simple_pdf):
    old, new = "o" * 32, "n" * 32
    secret = tmp_path / "api_key"
    secret.write_text(f"{old}\n{new}\n")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_KEY_FILE", str(secret))
    with TestClient(api_module.app) as c:
        files = {"file": ("s.pdf", simple_pdf.read_bytes())}
        assert c.post("/v1/extract?content=tables", files=files).status_code == 401
        assert c.post("/v1/extract?content=tables", files=files, headers={"X-API-Key": "wrong"}).status_code == 401
        for key in (old, new):
            assert c.post("/v1/extract?content=tables", files=files, headers={"X-API-Key": key}).status_code == 200
        assert c.get("/health").status_code == 200  # load-balancer probe stays open
