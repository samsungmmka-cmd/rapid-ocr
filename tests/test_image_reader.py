"""Image input: type detection, loading, quality notes, enhancement, OCR."""

import io
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest
from PIL import Image

from pdf_table_extractor import ImageDocumentResult, InvalidImageError, PDFTooLargeError, extract_document
from pdf_table_extractor.cli import EXIT_OK, EXIT_POOR_QUALITY, main
from pdf_table_extractor.image_enhance import correct_perspective
from pdf_table_extractor.image_quality import assess
from pdf_table_extractor.image_reader import ImageTableExtractor, _rotate, detect_orientation, load_frames
from pdf_table_extractor.reader import UnsupportedFileError, detect_type, iter_inputs

EXPECTED_ROWS = [["North", "1,200", "1,350"], ["South", "980", "1,010"]]


def render(pdf: Path, dpi: int = 200) -> np.ndarray:
    pix = pymupdf.open(pdf)[0].get_pixmap(dpi=dpi)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.stride)[:, : pix.w * 3].reshape(pix.h, pix.w, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def encode(img: np.ndarray, ext: str, jpeg_quality: int | None = None) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if jpeg_quality else []
    ok, buf = cv2.imencode(ext, img, params)
    assert ok
    return buf.tobytes()


@pytest.fixture
def page_img(simple_pdf) -> np.ndarray:
    return render(simple_pdf)


# ------------------------------------------------------------ type detection
def test_detect_type_by_content(simple_pdf, page_img):
    assert detect_type(simple_pdf.read_bytes()) == "pdf"
    for ext in (".png", ".jpg", ".tiff", ".bmp", ".webp"):
        assert detect_type(encode(page_img, ext)) == "image", ext
    with pytest.raises(UnsupportedFileError):
        detect_type(b"hello world")


def test_misnamed_file_is_routed_by_content(tmp_path, page_img):
    fake_pdf = tmp_path / "actually_png.pdf"
    fake_pdf.write_bytes(encode(page_img, ".png"))
    from pdf_table_extractor.reader import DocumentReader

    reader = DocumentReader()
    called = {}
    reader.images.extract = lambda src, frames=None: called.setdefault("image", src)
    reader.extract(fake_pdf)
    assert called["image"] == fake_pdf


def test_iter_inputs_expands_directories(tmp_path, simple_pdf, page_img):
    d = tmp_path / "in"
    (d / "sub").mkdir(parents=True)
    (d / "a.pdf").write_bytes(simple_pdf.read_bytes())
    (d / "b.png").write_bytes(encode(page_img, ".png"))
    (d / "notes.txt").write_text("x")
    (d / "sub" / "c.jpg").write_bytes(encode(page_img, ".jpg"))
    assert [p.name for p in iter_inputs([d])] == ["a.pdf", "b.png"]
    assert [p.name for p in iter_inputs([d], recursive=True)] == ["a.pdf", "b.png", "c.jpg"]


# ------------------------------------------------------------------ loading
def test_transparent_png_is_flattened_on_white():
    rgba = np.zeros((20, 30, 4), np.uint8)  # fully transparent black
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    [frame], fmt, _ = load_frames(buf.getvalue(), 10, 10_000_000)
    assert fmt == "PNG" and frame.min() == 255


def test_multi_page_tiff_and_dpi():
    frames = [Image.new("RGB", (40, 30), c) for c in ("white", "gray", "black")]
    buf = io.BytesIO()
    frames[0].save(buf, "TIFF", save_all=True, append_images=frames[1:], dpi=(300, 300))
    loaded, fmt, dpi = load_frames(buf.getvalue(), 10, 10_000_000)
    assert len(loaded) == 3 and fmt == "TIFF" and dpi == 300
    with pytest.raises(PDFTooLargeError):
        load_frames(buf.getvalue(), 2, 10_000_000)


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (60, 20), "white")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    [frame], _, _ = load_frames(buf.getvalue(), 10, 10_000_000)
    assert frame.shape[:2] == (60, 20)


def test_decompression_bomb_and_corrupt_input():
    buf = io.BytesIO()
    Image.new("L", (3000, 3000)).save(buf, "PNG")
    with pytest.raises(PDFTooLargeError):
        load_frames(buf.getvalue(), 10, max_pixels=1_000_000)
    with pytest.raises(InvalidImageError):
        load_frames(b"\x89PNG\r\n\x1a\n corrupted", 10, 10_000_000)


# ------------------------------------------------------------------ quality
def test_quality_flags(page_img):
    assert assess(page_img).issues == []
    h, w = page_img.shape[:2]
    blurry = cv2.GaussianBlur(page_img, (0, 0), 1.8)
    assert "blurry" in assess(blurry).issues
    tiny = cv2.resize(page_img, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
    assert "low_resolution" in assess(tiny).issues
    noisy = np.clip(page_img + np.random.default_rng(0).normal(0, 20, page_img.shape), 0, 255).astype(np.uint8)
    assert "noisy" in assess(noisy).issues
    faint = (page_img * 0.25 + 150).astype(np.uint8)
    assert "low_contrast" in assess(faint).issues
    shade = (page_img * (0.4 + 0.6 * np.linspace(0, 1, w))[None, :, None]).astype(np.uint8)
    report = assess(shade)
    assert "uneven_lighting" in report.issues
    assert all(isinstance(n, str) and n for n in report.notes)


def test_perspective_guard_and_correction(page_img):
    same, changed = correct_perspective(page_img)  # white page, table border only
    assert not changed and same is page_img
    h, w = page_img.shape[:2]
    canvas = np.full((h + 300, w + 300, 3), (50, 60, 70), np.uint8)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[180, 110], [w + 90, 160], [w + 230, h + 250], [60, h + 200]])
    photo = cv2.warpPerspective(page_img, cv2.getPerspectiveTransform(src, dst), (w + 300, h + 300), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
    flat, changed = correct_perspective(photo)
    assert changed
    assert flat.mean() > photo.mean() + 20  # desk removed


# ------------------------------------------------------------- real OCR ---
@pytest.mark.ocr
def test_orientation_detection(page_img):
    engine = ImageTableExtractor()._ocr
    assert detect_orientation(page_img, engine) == 0
    assert detect_orientation(cv2.rotate(page_img, cv2.ROTATE_90_COUNTERCLOCKWISE), engine) == 90
    assert detect_orientation(_rotate(page_img, 180), engine) == 180


@pytest.mark.ocr
def test_clean_image_end_to_end(tmp_path, page_img):
    path = tmp_path / "table.png"
    path.write_bytes(encode(page_img, ".png"))
    result = extract_document(path)
    assert isinstance(result, ImageDocumentResult)
    assert result.quality_level == "good"
    [t] = result.tables
    assert t.title == "Quarterly Revenue"
    assert t.header == ["Region", "Q1", "Q2"] and t.rows == EXPECTED_ROWS
    data = result.to_dict()
    assert data["input_type"] == "image" and data["quality"][0]["level"] == "good"


@pytest.mark.ocr
def test_unclear_photo_is_enhanced_and_noted(tmp_path, page_img):
    h, w = page_img.shape[:2]
    img = cv2.resize(page_img, (int(w * 0.55), int(h * 0.55)), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    rh, rw = img.shape[:2]
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D((rw / 2, rh / 2), 2.0, 1.0), (rw, rh), borderValue=(255, 255, 255))
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    path = tmp_path / "unclear.jpg"
    path.write_bytes(encode(img, ".jpg", jpeg_quality=70))

    result = ImageTableExtractor().extract(path)
    [t] = result.tables
    assert t.rows == EXPECTED_ROWS
    q = result.quality[0]
    assert {"rotated", "skewed"} <= set(q["issues"])
    assert any("rotated" in e for e in q["enhancements"])
    assert q["notes"]


@pytest.mark.ocr
def test_unreadable_image_is_reported_poor(tmp_path, page_img):
    h, w = page_img.shape[:2]
    img = cv2.resize(page_img, (w // 5, h // 5), interpolation=cv2.INTER_AREA)
    img = np.clip(img * 0.5 + 110, 0, 255).astype(np.uint8)
    path = tmp_path / "bad.jpg"
    path.write_bytes(encode(img, ".jpg", jpeg_quality=30))
    result = ImageTableExtractor().extract(path)
    assert result.quality_level == "poor"
    notes = " ".join(result.quality_notes())
    assert "Low resolution" in notes


@pytest.mark.ocr
@pytest.mark.slow
def test_cli_mixed_folder_and_strict_quality(tmp_path, simple_pdf, page_img):
    d = tmp_path / "in"
    d.mkdir()
    (d / "doc.pdf").write_bytes(simple_pdf.read_bytes())
    (d / "scan.png").write_bytes(encode(page_img, ".png"))
    out = tmp_path / "out"
    assert main([str(d), "-o", str(out), "-q"]) == EXIT_OK
    assert (out / "doc" / "doc.tables.json").exists()
    assert (out / "scan_png" / "scan.tables.json").exists()
    assert "Source image quality: good" in (out / "scan_png" / "scan.tables.md").read_text()

    h, w = page_img.shape[:2]
    bad = np.clip(cv2.resize(page_img, (w // 5, h // 5), interpolation=cv2.INTER_AREA) * 0.5 + 110, 0, 255).astype(np.uint8)
    bad_path = tmp_path / "bad.jpg"
    bad_path.write_bytes(encode(bad, ".jpg", jpeg_quality=30))
    assert main([str(bad_path), "--stdout", "-q"]) == EXIT_OK
    assert main([str(bad_path), "--stdout", "-q", "--strict-quality"]) == EXIT_POOR_QUALITY
