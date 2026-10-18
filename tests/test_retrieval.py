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


def test_vector_store_dense_search(vector_store):
    results = vector_store.search("حق الملكية", top_k=2)
    assert len(results) > 0
    assert results[0][0].article_number > 0


def test_vector_store_auto_seed(tmp_path, monkeypatch):
    import json

    from legal_rag.config import settings

    sample_json = tmp_path / "sample_corpus.json"
    sample_json.write_text(
        json.dumps(
            [
                {
                    "article_number": 1,
                    "text_ar": "نص المادة الأولى القانونية",
                    "citation": "Egyptian Civil Code, Article 1",
                    "book": "b",
                    "chapter": "c",
                    "section": "s",
                    "topic": "t",
                    "is_repealed": False,
                    "source_page": 1,
                    "paragraph_index": 0,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "PROCESSED_DATA_PATH", sample_json)
    store = LegalVectorStore(persist_dir=tmp_path / "seed_test_chroma")
    assert store.documents_indexed == 1
