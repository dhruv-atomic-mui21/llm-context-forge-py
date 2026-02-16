"""
Tests for ContextForge FastAPI Application
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import app


client = TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["version"] == "1.0.0"

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestTokenCountEndpoint:
    """Tests for token counting endpoint."""

    def test_count_tokens(self):
        response = client.post("/tokens/count", json={
            "text": "Hello world!",
            "model": "gpt-4o",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["tokens"] > 0
        assert data["model"] == "gpt-4o"
        assert data["context_window"] == 128000
        assert data["fits"] is True

    def test_count_empty_text(self):
        response = client.post("/tokens/count", json={
            "text": "",
            "model": "gpt-4o",
        })
        assert response.status_code == 200
        assert response.json()["tokens"] == 0


class TestTokenValidateEndpoint:
    """Tests for token validation endpoint."""

    def test_validate_fits(self):
        response = client.post("/tokens/validate", json={
            "text": "Hello",
            "max_tokens": 100,
            "model": "gpt-4o",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["fits"] is True
        assert data["remaining"] > 0

    def test_validate_too_long(self):
        response = client.post("/tokens/validate", json={
            "text": "word " * 100,
            "max_tokens": 5,
            "model": "gpt-4o",
        })
        assert response.status_code == 200
        assert response.json()["fits"] is False


class TestChunkEndpoint:
    """Tests for chunking endpoint."""

    def test_chunk_text(self):
        response = client.post("/chunk", json={
            "text": "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            "strategy": "paragraph",
            "max_tokens": 500,
            "model": "gpt-4o",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["num_chunks"] >= 1
        assert len(data["chunks"]) >= 1

    def test_chunk_invalid_strategy(self):
        response = client.post("/chunk", json={
            "text": "Hello",
            "strategy": "invalid_strategy",
        })
        assert response.status_code == 400


class TestContextAssembleEndpoint:
    """Tests for context assembly endpoint."""

    def test_assemble_blocks(self):
        response = client.post("/context/assemble", json={
            "blocks": [
                {"content": "System prompt", "priority": "CRITICAL", "label": "system"},
                {"content": "User context", "priority": "HIGH", "label": "context"},
            ],
            "max_tokens": 1000,
            "model": "gpt-4o",
        })
        assert response.status_code == 200
        data = response.json()
        assert "System prompt" in data["assembled"]
        assert data["usage"]["num_blocks"] == 2


class TestCompressEndpoint:
    """Tests for compression endpoint."""

    def test_compress_text(self):
        long_text = "This is a sentence. " * 50
        response = client.post("/compress", json={
            "text": long_text,
            "target_tokens": 20,
            "strategy": "extractive",
            "model": "gpt-4o",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["compressed_tokens"] <= 25  # some tolerance
        assert data["savings_pct"] > 0

    def test_compress_invalid_strategy(self):
        response = client.post("/compress", json={
            "text": "Hello",
            "target_tokens": 100,
            "strategy": "invalid",
        })
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
