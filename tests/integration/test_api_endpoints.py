"""
Tests for LayoutLM Forge FastAPI Application
"""

import pytest
from fastapi.testclient import TestClient

from layoutlm_forge.api.app import app

client = TestClient(app)

class TestHealthEndpoints:
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "providers" in data

class TestTokenCountEndpoint:
    def test_count_tokens(self):
        response = client.post("/api/v1/tokens/count", json={
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
        response = client.post("/api/v1/tokens/count", json={
            "text": "",
            "model": "gpt-4o",
        })
        assert response.status_code == 200
        assert response.json()["tokens"] == 0

class TestTokenValidateEndpoint:
    def test_validate_fits(self):
        response = client.post("/api/v1/tokens/validate", json={
            "text": "Hello",
            "max_tokens": 100,
            "model": "gpt-4o",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["fits"] is True
        assert data["remaining"] > 0

    def test_validate_too_long(self):
        response = client.post("/api/v1/tokens/validate", json={
            "text": "word " * 100,
            "max_tokens": 5,
            "model": "gpt-4o",
        })
        assert response.status_code == 200
        assert response.json()["fits"] is False

class TestChunkEndpoint:
    def test_chunk_text(self):
        response = client.post("/api/v1/chunks/", json={
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
        response = client.post("/api/v1/chunks/", json={
            "text": "Hello",
            "strategy": "invalid_strategy",
        })
        assert response.status_code == 400

class TestContextAssembleEndpoint:
    def test_assemble_blocks(self):
        response = client.post("/api/v1/context/assemble", json={
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
    def test_compress_text(self):
        long_text = "This is a sentence. " * 50
        response = client.post("/api/v1/compress/", json={
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
        response = client.post("/api/v1/compress/", json={
            "text": "Hello",
            "target_tokens": 100,
            "strategy": "invalid",
        })
        assert response.status_code == 400

class TestCostEndpoint:
    def test_estimate_cost(self):
        response = client.post("/api/v1/cost/estimate", json={
            "text": "Hello world",
            "model": "gpt-4o"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["input_tokens"] > 0
        assert data["input_cost_usd"] > 0
        assert data["model"] == "gpt-4o"
