from pdf_table_extractor.normalize import clean_grid, clean_grid_indexed, clean_text, fix_ocr_text, looks_like_page_number


def test_clean_text_whitespace_and_invisibles():
    assert clean_text(None) == ""
    assert clean_text("  a b\n c​ ") == "a b c"
    assert clean_text("ﬁle") == "file"
    assert clean_text("soft­hyphen") == "softhyphen"


def test_clean_grid_drops_empty_and_pads():
    grid = [["a", None, ""], ["", "", ""], ["b", "", "x"], ["c"]]
    assert clean_grid(grid) == [["a", ""], ["b", "x"], ["c", ""]]


def test_clean_grid_indexed_reports_kept_indices():
    grid, rows, cols = clean_grid_indexed([["", "", ""], ["a", "", "b"]])
    assert grid == [["a", "b"]]
    assert rows == [1]
    assert cols == [0, 2]
    assert clean_grid_indexed([]) == ([], [], [])


def test_fix_ocr_text_only_touches_pure_dash_cells():
    assert fix_ocr_text("一") == "—"
    assert fix_ocr_text("ー") == "—"
    assert fix_ocr_text("一二") == "一二"  # real CJK text is untouched
    assert fix_ocr_text("5 GB") == "5 GB"


def test_page_number_detection():
    for s in ["3", "Page 3", "page 3 of 10", "3/10"]:
        assert looks_like_page_number(s)
    assert not looks_like_page_number("3. Results")
