"""API smoke tests (no Chroma required for health/taxonomies)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["version"] == "4.0.0"
    assert body["resolver"] == "jev"


def test_taxonomies():
    res = client.get("/api/taxonomies")
    assert res.status_code == 200
    body = res.json()
    assert "taxonomies" in body or "pairs" in body or isinstance(body, dict)


def test_models():
    res = client.get("/api/models")
    assert res.status_code == 200
    assert "P" in res.json().get("models", [])
