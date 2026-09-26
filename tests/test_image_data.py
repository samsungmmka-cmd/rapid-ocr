"""ImageDataExtractor: unit tests for each stage + regression on real invoices."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from pdf_table_extractor import ImageDataConfig, ImageDataExtractor
from pdf_table_extractor.cli import EXIT_OK, main
from pdf_table_extractor.image_data import _apply, _rot_matrix
from pdf_table_extractor.image_layout import Item, inline_kv, normalize_value, pair_kvs, parse_number, xy_cut
from pdf_table_extractor.image_tables import (
    _split_band,
    find_borderless_tables,
    is_header_cell,
    split_line_at,
)
from pdf_table_extractor.ocr_models import Candidate, OCRLine, OCRModels, _adjusted

DATA = Path(__file__).resolve().parent.parent / "data"
_ids = iter(range(10_000))


def line(text, x0, y0, x1, y1, score=0.99, words=None):
    poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    return OCRLine(id=next(_ids), poly=poly, text=text, score=score, words=words or [])


# ------------------------------------------------------------------ values
def test_parse_number_and_normalize():
    assert parse_number("$1,234.50") == 1234.5
    assert parse_number("7226-") == 7226.0
    assert parse_number("Rs. 450") == 450.0
    assert parse_number("11/02/2019") is None
    assert parse_number("US-001") is None
    assert normalize_value("28|10|2024") == "28/10/2024"
    assert normalize_value("Bvc Logistics") == "Bvc Logistics"


def test_inline_key_values():
    kv = inline_kv(line("Date: 28|10|2024", 0, 0, 100, 10))
    assert (kv.key, kv.value) == ("Date", "28/10/2024")
    kv = inline_kv(line("PAN No. DODPS7841F", 0, 0, 100, 10))
    assert (kv.key, kv.value) == ("PAN No.", "DODPS7841F")
    assert inline_kv(line("Please make checks payable to: East Repair Inc.", 0, 0, 300, 10)) is None


def test_side_by_side_pairs():
    lines = [
        line("INVOICE #", 460, 200, 520, 215),
        line("US-001", 650, 201, 696, 214),
        line("BILL TO", 50, 200, 100, 215),  # label with a label on its right: no pair
        line("SHIP TO", 260, 200, 310, 215),
        line("BIG BRAND", 100, 20, 500, 70),  # big heading next to a small phone number
        line("90224 18585", 520, 30, 600, 42),
        line("OK", 300, 600, 320, 615, score=0.6),  # scribble
        line("0+", 330, 600, 350, 615, score=0.7),
    ]
    kvs, used = pair_kvs(lines, page_width=750)
    assert [(k.key, k.value) for k in kvs] == [("INVOICE #", "US-001")]


def test_header_cells():
    assert is_header_cell("UNIT PRICE") and is_header_cell("QTY")
    assert not is_header_cell("2 Court Square")
    assert not is_header_cell("Front and rear brake cables")


# ------------------------------------------------------------------ tables
def test_split_line_uses_word_boxes():
    words = [("3", np.array([[180, 0], [195, 0], [195, 10], [180, 10]], float)),
             ("Labor", np.array([[280, 0], [330, 0], [330, 10], [280, 10]], float)),
             ("3hrs", np.array([[335, 0], [370, 0], [370, 10], [335, 10]], float))]
    ln = line("3 Labor 3hrs", 180, 0, 370, 10, words=words)
    left, right = split_line_at(ln, 300)  # proportional split would cut after 'Labor'
    assert (left.text, right.text) == ("3", "Labor 3hrs")


def test_borderless_table_with_footer_fields():
    rows = [
        [("QTY", 70, 97), ("DESCRIPTION", 225, 303), ("UNIT PRICE", 457, 522), ("AMOUNT", 634, 686)],
        [("1", 80, 90), ("Front and rear brake cables", 128, 283), ("100.00", 485, 522), ("100.00", 648, 686)],
        [("2", 80, 90), ("New set of pedal arms", 128, 254), ("15.00", 490, 522), ("30.00", 653, 686)],
        [("Subtotal", 476, 522), ("145.00", 648, 686)],
        [("TOTAL", 481, 522), ("$154.06", 628, 686)],
    ]
    lines = [line(t, a, 350 + 33 * r, b, 362 + 33 * r) for r, row in enumerate(rows) for t, a, b in row]
    [table] = find_borderless_tables(lines, char_h=10)
    texts = table.texts()
    assert texts[0] == ["QTY", "DESCRIPTION", "UNIT PRICE", "AMOUNT"]
    assert texts[1] == ["1", "Front and rear brake cables", "100.00", "100.00"]
    assert len(texts) == 3
    assert [(a.text, b.text) for a, b in table.footer_kv] == [("Subtotal", "145.00"), ("TOTAL", "$154.06")]


def test_handwritten_rows_split_by_item_numbers():
    band = []
    for k in range(3):
        y = 100 + 60 * k
        band += [(line(f"{k + 1})", 10, y, 30, y + 40), 0), (line(f"item {k}", 60, y, 200, y + 40), 1), (line(f"{(k + 1) * 100}-", 400, y - 12, 450, y + 25), 2)]
    band.append((line("continued text", 60, 125, 200, 150), 1))  # wraps under item 1
    band.append((line("stamp", 300, 400, 380, 440, score=0.5), 1))  # far below: overflow
    rows, overflow = _split_band(band, 3, top=90, char_h=20)
    assert len(rows) == 3
    first = {c: [ln.text for ln, cc in rows[0] if cc == c] for c in range(3)}
    assert first == {0: ["1)"], 1: ["item 0", "continued text"], 2: ["100-"]}
    assert [ln.text for ln in overflow] == ["stamp"]


def test_header_band_is_not_split():
    band = [(line("AMOUNT", 400, 10, 470, 30), 0), (line("Rs.", 400, 35, 430, 55), 0), (line("P.", 480, 35, 500, 55), 1)]
    rows, overflow = _split_band(band, 2, top=0, char_h=20)
    assert len(rows) == 1 and not overflow


# ---------------------------------------------------------- reading order
def test_xy_cut_reads_columns_before_rows():
    def it(name, x0, y0, x1, y1):
        return Item((x0, y0, x1, y1), "line", name)

    items = [it("BILL TO", 50, 200, 100, 215), it("John", 50, 225, 100, 240), it("SHIP TO", 260, 200, 310, 215),
             it("Jane", 260, 225, 310, 240), it("Footer", 50, 400, 300, 415)]
    order = [i.ref for leaf in xy_cut(items, lh=15) for i in leaf]
    assert order == ["BILL TO", "John", "SHIP TO", "Jane", "Footer"]


# --------------------------------------------------------------- ensemble
def test_candidate_selection_rules():
    base = Candidate("-9乙", 0.70, "small")
    fixed = Candidate("7226-", 0.92, "en")
    assert _adjusted(fixed, [base, fixed], base.text, True) > _adjusted(base, [base, fixed], base.text, True)
    long = Candidate("Rupees in Word: seven thousand", 0.91, "small")
    trunc = Candidate("Rau", 0.95, "medium")
    assert _adjusted(long, [long, trunc], long.text, True) > _adjusted(trunc, [long, trunc], long.text, True)


def test_missing_models_degrade_gracefully(monkeypatch):
    models = OCRModels()
    real = models._params

    def params(name):
        if name != "small":
            raise RuntimeError("offline")
        return real(name)

    monkeypatch.setattr(models, "_params", params)
    img = np.full((60, 200, 3), 255, np.uint8)
    cv2.putText(img, "Total 450", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    assert models.recognize_crop(img, "medium") is None
    assert set(models.unavailable) == {"medium"}


def test_rotation_matrix_matches_cv2():
    img = np.zeros((30, 50), np.uint8)
    img[5, 7] = 255
    for angle, code in ((90, cv2.ROTATE_90_CLOCKWISE), (180, cv2.ROTATE_180), (270, cv2.ROTATE_90_COUNTERCLOCKWISE)):
        y, x = np.argwhere(cv2.rotate(img, code) == 255)[0]
        assert np.allclose(_apply(_rot_matrix(angle, 50, 30), [[7, 5]])[0], [x, y])


def test_config_validation():
    with pytest.raises(ValueError):
        ImageDataConfig(accuracy="perfect")


# ------------------------------------------------------ real documents ----
@pytest.fixture(scope="module")
def extractor():
    return ImageDataExtractor()


@pytest.mark.ocr
@pytest.mark.slow
def test_printed_invoice(extractor):
    if not (DATA / "invoice1.png").exists():
        pytest.skip("data/invoice1.png not present")
    r = extractor.extract(DATA / "invoice1.png")
    fields = {kv.key: kv.value for kv in r.key_values}
    assert fields == {
        "INVOICE #": "US-001",
        "INVOICE DATE": "11/02/2019",
        "P.O.#": "2312/2019",
        "DUE DATE": "26/02/2019",
        "Subtotal": "145.00",
        "Sales Tax 6.25%": "9.06",
        "TOTAL": "$154.06",
    }
    assert r.get("total") == "$154.06" and r.key_values[2].number == 154.06
    [t] = r.tables
    assert t.header == ["QTY", "DESCRIPTION", "UNIT PRICE", "AMOUNT"]
    assert t.rows == [
        ["1", "Front and rear brake cables", "100.00", "100.00"],
        ["2", "New set of pedal arms", "15.00", "30.00"],
        ["3", "Labor 3hrs", "5.00", "15.00"],
    ]
    for expected in ["East Repair Inc.", "3787 Pineview Drive", "Cambridge, MA 12210", "TERMS & CONDITIONS",
                     "Payment is due within 15 days", "Please make checks payable to: East Repair Inc."]:
        assert expected in r.text
    assert r.text.index("SHIP TO") < r.text.index("INVOICE #") < r.text.index("QTY")
    assert "low_resolution" in r.pages[0].quality["issues"]


@pytest.mark.ocr
@pytest.mark.slow
def test_handwritten_invoice(extractor):
    if not (DATA / "hw.jpg").exists():
        pytest.skip("data/hw.jpg not present")
    r = extractor.extract(DATA / "hw.jpg")
    assert r.file_format == "WEBP"  # misnamed file, detected by content
    fields = {kv.key.replace(" ", ""): kv.value for kv in r.key_values}
    assert fields["BillNo."] == "287"
    assert fields["Date"] == "28/10/2024"
    assert fields["Total"].startswith("7226")
    assert fields["PANNo."] == "DODPS7841F"
    [t] = r.tables
    assert t.header[:2] == ["Sr. No", "PARTICULARS"] and len(t.header) == 7
    assert len(t.rows) == 9
    amounts = [row[5].rstrip("- ").strip() for row in t.rows]
    assert amounts == ["240", "550", "460", "900", "390", "2300", "836", "750", "800"]
    rates = [row[4] for row in t.rows]
    assert sum(a == b for a, b in zip(rates, ["240", "550", "230", "450", "195", "2300", "836", "750", "800"])) >= 8
    assert r.pages[0].review(0.8), "handwriting should produce review items"
    assert r.quality_level in {"fair", "poor"}


@pytest.mark.ocr
@pytest.mark.slow
def test_coordinates_map_back_through_rotation(tmp_path):
    if not (DATA / "invoice1.png").exists():
        pytest.skip("data/invoice1.png not present")
    fast = ImageDataExtractor(ImageDataConfig(accuracy="fast"))
    src = cv2.cvtColor(np.asarray(Image.open(DATA / "invoice1.png").convert("RGB")), cv2.COLOR_RGB2BGR)
    rotated = tmp_path / "rot.png"
    cv2.imwrite(str(rotated), cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE))
    a, b = fast.extract(DATA / "invoice1.png"), fast.extract(rotated)
    la = next(ln for ln in a.pages[0].lines if ln.text == "US-001")
    lb = next(ln for ln in b.pages[0].lines if ln.text == "US-001")
    h, w = src.shape[:2]
    expected = _apply(_rot_matrix(270, w, h), la.poly)  # original -> CCW-rotated image
    assert np.abs(np.sort(expected, axis=0) - np.sort(lb.poly, axis=0)).max() < 6


@pytest.mark.ocr
def test_pdf_input_and_cli(tmp_path, simple_pdf):
    r = ImageDataExtractor(ImageDataConfig(accuracy="fast")).extract(simple_pdf)
    assert r.input_type == "pdf"
    [t] = r.tables
    assert t.header == ["Region", "Q1", "Q2"] and t.rows[0] == ["North", "1,200", "1,350"]
    assert "Quarterly Revenue" in r.text
    out = tmp_path / "out"
    assert main([str(simple_pdf), "--content", "all", "--accuracy", "fast", "-o", str(out), "-q"]) == EXIT_OK
    assert sorted(p.name for p in out.iterdir()) == ["simple.data.json", "simple.data.md", "simple.txt", "simple_table01.csv"]


@pytest.mark.ocr
@pytest.mark.slow
def test_invoice_pdf_all_fields(extractor):
    if not (DATA / "invoice.pdf").exists():
        pytest.skip("data/invoice.pdf not present")
    r = extractor.extract(DATA / "invoice.pdf")
    fields = {kv.key: kv.value for kv in r.key_values}
    assert fields == {
        "Invoice Number": "INV-3337",
        "Order Number": "12345",
        "Invoice Date": "January 25, 2016",
        "Due Date": "January 31, 2016",
        "Total Due": "$93.50",
        "Sub Total": "$85.00",
        "Tax": "$8.50",
        "Total": "$93.50",
        "ACC #": "1234 1234",
        "BSB #": "4321 432",
    }
    [t] = r.tables
    assert t.rows[0][0] == "1.00" and t.rows[0][-1] == "$85.00"
    assert "123 Somewhere Street" in r.text and "Melbourne, VIC 3000" in r.text
