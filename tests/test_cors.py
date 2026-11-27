"""CORS is opt-in: off by default, exact-origin when RAG_CORS_ORIGINS is set."""
import importlib

from fastapi.testclient import TestClient


def _fresh_app(monkeypatch, origins: str | None):
    if origins is None:
        monkeypatch.delenv("RAG_CORS_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("RAG_CORS_ORIGINS", origins)
    import app.config
    import app.main

    app.config.get_settings.cache_clear()
    return importlib.reload(app.main).app


def _preflight(client, origin):
    return client.options(
        "/query",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )


def test_no_cross_origin_access_by_default(monkeypatch):
    client = TestClient(_fresh_app(monkeypatch, None))
    response = _preflight(client, "https://evil.example")
    assert "access-control-allow-origin" not in response.headers


def test_configured_origin_is_allowed_and_others_are_not(monkeypatch):
    client = TestClient(_fresh_app(monkeypatch, '["https://harshitgadge.github.io"]'))
    ok = _preflight(client, "https://harshitgadge.github.io")
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "https://harshitgadge.github.io"
    other = _preflight(client, "https://evil.example")
    assert "access-control-allow-origin" not in other.headers
    _fresh_app(monkeypatch, None)  # leave a default app behind for other tests
