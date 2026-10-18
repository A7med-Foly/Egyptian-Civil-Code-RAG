"""Pytest fixtures for Arabic Legal RAG test suite."""

import pytest
from starlette.testclient import TestClient

from legal_rag.api.main import app


@pytest.fixture(scope="session")
def sample_article():
    """Sample article fixture (Article 147 - Contract Law)."""
    return {
        "article_number": 147,
        "book": "Obligations or Personal Rights",
        "chapter": "Sources of Obligations",
        "section": "Contracts",
        "topic": "The Effects of a Contract",
        "text_ar": "العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين...",
        "text_en": "The contract makes the law of the parties...",
        "is_repealed": False,
        "source_page": 16,
        "citation": "Egyptian Civil Code, Article 147",
    }


@pytest.fixture(scope="session")
def sample_repealed_article():
    """Sample repealed article fixture (Article 54)."""
    return {
        "article_number": 54,
        "book": "General Provisions",
        "chapter": "Repealed Section",
        "section": "",
        "topic": "Repealed by Law",
        "text_ar": "ألغيت المواد من 54 إلى 80 بموجب القرار الجمهوري بالقانون رقم 384 لسنة 1956.",
        "text_en": "Articles 54 to 80 were repealed by Presidential Decree Law No. 384 of 1956.",
        "is_repealed": True,
        "source_page": 7,
        "citation": "Egyptian Civil Code, Article 54",
    }


@pytest.fixture(scope="session")
def client():
    """Test client with lifespan context to ensure model is loaded once."""
    with TestClient(app) as test_client:
        yield test_client
