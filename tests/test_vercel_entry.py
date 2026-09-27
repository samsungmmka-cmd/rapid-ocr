"""The Vercel entrypoint (app.py) must expose the full API unchanged."""

import pytest
from fastapi.testclient import TestClient

import app as entry


def test_entrypoint_serves_api(simple_pdf):
    with TestClient(entry.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert c.get("/docs").status_code == 200
        r = c.post("/v1/extract?content=tables", files={"file": ("s.pdf", simple_pdf.read_bytes())})
        assert r.status_code == 200 and r.json()["n_tables"] == 1


def test_entrypoint_is_fail_closed_in_production(monkeypatch, simple_pdf):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("API_KEY_FILE", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        with TestClient(entry.app):
            pass
    key = "k" * 32
    monkeypatch.setenv("API_KEY", key)
    with TestClient(entry.app) as c:
        files = {"file": ("s.pdf", simple_pdf.read_bytes())}
        assert c.post("/v1/extract?content=tables", files=files).status_code == 401
        assert c.post("/v1/extract?content=tables", files=files, headers={"X-API-Key": key}).status_code == 200


def test_upload_limit_still_applies(monkeypatch):
    import pdf_table_extractor.api as api

    monkeypatch.setattr(api, "MAX_UPLOAD_MB", 0.01)
    with TestClient(entry.app) as c:
        r = c.post("/v1/extract", files={"file": ("big.pdf", b"%PDF-" + b"0" * 200_000)})
        assert r.status_code == 413
