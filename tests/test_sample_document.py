"""End-to-end checks on the bundled SLT-Mobitel sample (package.pdf)."""

import pytest

from pdf_table_extractor import ExtractionConfig, extract_tables

EXPECTED_TITLES = [
    "1. Prepaid / Combo Packages",
    "2. Anytime Data Plans",
    "3. Mobitel 1598 Digital Super Combo",
    "4. Mobitel 1278 / 368 Packages",
    "5. Work From Home Packages",
    "6. Work & Learn Add-on",
    "7. Postpaid Broadband Plans",
    "8. 5G Postpaid Base Packages",
]
EXPECTED_SHAPES = [(7, 5), (11, 4), (1, 7), (2, 7), (3, 4), (2, 4), (4, 6), (3, 6)]


def test_native_sample(sample_pdf):
    result = extract_tables(sample_pdf)
    assert result.ok
    assert [t.title for t in result.tables] == EXPECTED_TITLES
    assert [(t.n_rows, t.n_cols) for t in result.tables] == EXPECTED_SHAPES
    assert all(t.method == "native" for t in result.tables)
    first = result.tables[0].records()[0]
    assert first == {"Price": "Rs.187", "Voice": "100 any-net", "SMS": "100 any-net", "Anytime Data": "1.5 GB", "Validity": "7 days"}
    assert result.tables[7].rows[0][4] == "—"


@pytest.mark.ocr
@pytest.mark.slow
def test_forced_ocr_matches_native(sample_pdf):
    native = extract_tables(sample_pdf, ExtractionConfig(mode="native"))
    ocr = extract_tables(sample_pdf, ExtractionConfig(mode="ocr"))
    assert [(t.n_rows, t.n_cols) for t in ocr.tables] == EXPECTED_SHAPES
    total = mismatched = 0
    for a, b in zip(native.tables, ocr.tables):
        for ra, rb in zip([a.header] + a.rows, [b.header] + b.rows):
            for x, y in zip(ra, rb):
                total += 1
                mismatched += x != y
    assert mismatched / total <= 0.02, f"{mismatched}/{total} cells differ"
