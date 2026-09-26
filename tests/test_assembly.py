from pdf_table_extractor.assembly import assemble
from pdf_table_extractor.config import ExtractionConfig
from pdf_table_extractor.layout import PageData, RawTable, TextLine

CFG = ExtractionConfig()


def page(no, tables, lines=()):
    return PageData(page=no, width=600, height=800, tables=list(tables), lines=list(lines))


def raw(no, bbox, grid, **kw):
    return RawTable(page=no, bbox=bbox, grid=grid, method="native", **kw)


def test_title_is_nearest_line_above():
    t = raw(1, (100, 200, 400, 300), [["H1", "H2"], ["a", "b"]])
    lines = [TextLine((40, 100, 200, 112), "Intro text"), TextLine((40, 175, 200, 190), "Table heading")]
    [out] = assemble([page(1, [t], lines)], CFG)
    assert out.title == "Table heading"
    assert out.header == ["H1", "H2"]
    assert out.rows == [["a", "b"]]


def test_title_too_far_or_page_number_is_ignored():
    t = raw(1, (100, 200, 400, 300), [["H1", "H2"], ["a", "b"]])
    lines = [TextLine((40, 20, 200, 30), "Far away"), TextLine((40, 185, 60, 195), "12")]
    [out] = assemble([page(1, [t], lines)], CFG)
    assert out.title is None


def test_dangling_heading_from_previous_page():
    p1 = page(1, [], [TextLine((40, 760, 300, 775), "4. Next Section")])
    t = raw(2, (100, 30, 400, 90), [["H1", "H2"], ["a", "b"]])
    [out] = assemble([p1, page(2, [t])], CFG)
    assert out.title == "4. Next Section"
    assert "title taken from bottom of page 1" in out.warnings


def test_continuation_with_repeated_header_is_merged():
    a = raw(1, (40, 80, 330, 780), [["Item", "Qty"], ["x", "1"]])
    b = raw(2, (40, 30, 330, 90), [["Item", "Qty"], ["y", "2"]])
    [out] = assemble([page(1, [a], [TextLine((40, 60, 200, 74), "Parts")]), page(2, [b])], CFG)
    assert out.pages == [1, 2]
    assert out.rows == [["x", "1"], ["y", "2"]]
    assert out.title == "Parts"


def test_continuation_without_header_merges_when_aligned():
    a = raw(1, (40, 80, 330, 780), [["Item", "Qty"], ["x", "1"]])
    b = raw(2, (41, 30, 331, 90), [["y", "2"], ["z", "3"]])
    [out] = assemble([page(1, [a]), page(2, [b])], CFG)
    assert out.rows == [["x", "1"], ["y", "2"], ["z", "3"]]


def test_no_merge_when_new_table_has_a_title_or_different_shape():
    a = raw(1, (40, 80, 330, 780), [["Item", "Qty"], ["x", "1"]])
    b = raw(2, (40, 30, 330, 90), [["Item", "Qty"], ["y", "2"]])
    heading = TextLine((40, 10, 200, 22), "Other table")
    assert len(assemble([page(1, [a]), page(2, [b], [heading])], CFG)) == 2

    c = raw(2, (40, 30, 330, 90), [["A", "B", "C"], ["1", "2", "3"]])
    assert len(assemble([page(1, [a]), page(2, [c])], CFG)) == 2

    cfg = ExtractionConfig(merge_continued_tables=False)
    assert len(assemble([page(1, [a]), page(2, [b])], cfg)) == 2


def test_multi_row_header_and_low_confidence_remap():
    t = raw(
        1,
        (0, 0, 100, 100),
        [["Price", "Price"], ["Min", "Max"], ["1", "2"]],
        low_confidence_cells=[(1, 0, 0.4), (2, 1, 0.5)],
    )
    [out] = assemble([page(1, [t])], ExtractionConfig(header_rows=2))
    assert out.header == ["Price / Min", "Price / Max"]
    assert out.rows == [["1", "2"]]
    assert out.low_confidence_cells == [
        {"section": "header", "row": 1, "col": 0, "score": 0.4},
        {"section": "body", "row": 0, "col": 1, "score": 0.5},
    ]
