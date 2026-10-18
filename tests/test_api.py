"""API integration tests for Arabic Legal Document Q&A endpoints."""


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
