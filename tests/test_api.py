"""API integration tests for Arabic Legal Document Q&A endpoints."""

import pytest


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["documents_indexed"] >= 1149


def test_metadata_endpoint(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["api_version"] == "0.1.0"
    assert "Egyptian Civil Code" in data["corpus_name"]
    assert data["documents_indexed"] >= 1149


def test_ask_happy_path(client):
    payload = {
        "question": "ما هي القوة الملزمة للعقد؟",
        "top_k": 3,
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "sources" in data
    assert isinstance(data["sources"], list)
    for src in data["sources"]:
        assert src.startswith("Egyptian Civil Code, Article")

    # Assert correlation ID is present and matches response header
    header_corr_id = response.headers.get("X-Request-ID")
    assert header_corr_id is not None
    assert data["correlation_id"] == header_corr_id
    assert data["latency_ms"] > 0


def test_ask_empty_question_returns_422(client):
    # Sending empty whitespace string must be rejected
    payload = {"question": "   "}
    response = client.post("/ask", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Unprocessable Entity"
    assert "correlation_id" in data


def test_ask_stream_endpoint(client):
    payload = {
        "question": "ما حكم الشرط الواقف؟",
        "top_k": 2,
    }
    response = client.post("/ask/stream", json=payload)
    assert response.status_code == 200
    assert len(response.text) > 0


def test_generator_mock_behaviors():
    from legal_rag.rag.generator import LegalGenerator

    gen = LegalGenerator(provider="mock")
    # 1. Repealed article prompt
    rep_res = gen._generate_mock("سؤال عن مادة ملغاة Egyptian Civil Code, Article 54")
    assert "ملغاة" in rep_res
    assert "Article 54" in rep_res

    # 2. Normal cited prompt
    cited_res = gen._generate_mock("سؤال عن العقد Egyptian Civil Code, Article 147")
    assert "Article 147" in cited_res

    # 3. No citations prompt
    no_cite_res = gen._generate_mock("سؤال عام بدون أي سند قانوني")
    assert "لا يتضمن القانون المدني المصري" in no_cite_res
    assert "تنويه" in no_cite_res


@pytest.mark.asyncio
async def test_generator_async_mock():
    from legal_rag.rag.generator import LegalGenerator

    gen = LegalGenerator(provider="mock")
    resp = await gen.generate_response("Egyptian Civil Code, Article 10", "sys")
    assert "Article 10" in resp
    tokens = [tok async for tok in gen.generate_stream("Egyptian Civil Code, Article 10", "sys")]
    assert len(tokens) > 0
