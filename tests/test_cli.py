import json

import pytest

from pdf_table_extractor.cli import EXIT_INPUT, EXIT_OK, EXIT_PARTIAL, _worst, main, parse_pages


def test_parse_pages():
    assert parse_pages("1-3,5, 3") == [1, 2, 3, 5]
    for bad in ["", "0", "3-1", "a"]:
        with pytest.raises(Exception):
            parse_pages(bad)


def test_worst_exit_code_ordering():
    assert _worst(EXIT_PARTIAL, EXIT_INPUT) == EXIT_INPUT
    assert _worst(EXIT_OK, EXIT_PARTIAL) == EXIT_PARTIAL


def test_cli_writes_all_formats(simple_pdf, tmp_path):
    out = tmp_path / "out"
    assert main([str(simple_pdf), "-o", str(out), "-q"]) == EXIT_OK
    names = sorted(p.name for p in out.iterdir())
    assert names == ["simple.tables.json", "simple.tables.md", "simple_table01_quarterly-revenue.csv"]
    csv_text = (out / "simple_table01_quarterly-revenue.csv").read_text()
    assert csv_text.splitlines()[0] == "Region,Q1,Q2"
    assert '"1,200"' in csv_text
    data = json.loads((out / "simple.tables.json").read_text())
    assert data["tables"][0]["records"][0]["Region"] == "North"


def test_cli_stdout_json(simple_pdf, capsys):
    assert main([str(simple_pdf), "--stdout", "-q"]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["n_tables"] == 1


def test_cli_bad_input_exit_code(tmp_path):
    bad = tmp_path / "x.pdf"
    bad.write_text("nope")
    assert main([str(bad), "-q", "-o", str(tmp_path)]) == EXIT_INPUT


def test_cli_password_env(simple_pdf, tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        main([str(simple_pdf), "--password-env", "UNSET_VAR_FOR_TEST", "--stdout"])


def test_cli_partial_exit_code(continued_pdf, tmp_path, monkeypatch):
    import pdf_table_extractor.pipeline as pipeline

    def boom(*a, **k):
        raise RuntimeError("fail")

    monkeypatch.setattr(pipeline, "extract_native", boom)
    assert main([str(continued_pdf), "-q", "-o", str(tmp_path)]) == EXIT_PARTIAL
