from pdf_table_extractor.models import DocumentResult, PageResult, Table


def _table(**kw):
    base = dict(page=1, index=0, bbox=(0, 0, 10, 10), method="native", header=["A", "", "A"], rows=[["1", "2", "3"], ["4"]])
    base.update(kw)
    return Table(**base)


def test_column_keys_are_unique_and_non_empty():
    assert _table().column_keys() == ["A", "column_2", "A_2"]


def test_records_pad_short_rows():
    assert _table().records()[1] == {"A": "4", "column_2": "", "A_2": ""}


def test_document_status_and_serialisation():
    doc = DocumentResult(source="x.pdf", sha256="0", n_pages=2, mode="auto", tables=[_table()])
    doc.pages = [PageResult(page=1, method="native"), PageResult(page=2, method="error", error="boom")]
    data = doc.to_dict()
    assert data["status"] == "partial"
    assert data["failed_pages"] == [2]
    assert data["tables"][0]["records"][0] == {"A": "1", "column_2": "2", "A_2": "3"}
    assert data["schema_version"] == "1.0"
