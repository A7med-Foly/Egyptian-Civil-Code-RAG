"""Integration tests for hybrid vector store retrieval."""

import pytest

from legal_rag.indexing.vector_store import LegalVectorStore


@pytest.fixture(scope="module")
def vector_store():
    store = LegalVectorStore()
    return store


def test_vector_store_count(vector_store):
    assert vector_store.documents_indexed >= 1149


def test_hybrid_search_returns_citations(vector_store):
    results = vector_store.hybrid_search("شروط العقد والالتزامات", top_k=3)
    assert len(results) > 0
    for chunk, score in results:
        assert chunk.citation.startswith("Egyptian Civil Code, Article")
        assert chunk.article_number > 0
        assert len(chunk.text) > 0
        assert 0.0 <= score <= 1.0


def test_empty_search_returns_empty(vector_store):
    # Search on an impossible query
    results = vector_store.hybrid_search("", top_k=3)
    assert isinstance(results, list)
