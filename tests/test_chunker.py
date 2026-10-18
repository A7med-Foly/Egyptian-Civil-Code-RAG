"""Unit tests for article-aware legal chunker."""

from legal_rag.indexing.chunker import ArticleChunker, LegalChunk


def test_chunk_single_article(sample_article):
    chunker = ArticleChunker(max_chars_per_chunk=2000)
    chunks = chunker.chunk_article(sample_article)
    assert len(chunks) == 1
    c = chunks[0]
    assert isinstance(c, LegalChunk)
    assert c.article_number == 147
    assert c.citation == "Egyptian Civil Code, Article 147"
    assert c.is_repealed is False
    assert "العقد شريعة المتعاقدين" in c.text


def test_chunk_repealed_article(sample_repealed_article):
    chunker = ArticleChunker()
    chunks = chunker.chunk_article(sample_repealed_article)
    assert len(chunks) == 1
    assert chunks[0].is_repealed is True
    assert chunks[0].citation == "Egyptian Civil Code, Article 54"


def test_chunk_long_article_splits_paragraphs():
    long_article = {
        "article_number": 999,
        "citation": "Egyptian Civil Code, Article 999",
        "book": "Book Test",
        "chapter": "Chapter Test",
        "section": "Section Test",
        "topic": "Topic Test",
        "text_ar": "الفقرة الأولى من المادة الطويلة.\n\nالفقرة الثانية من المادة الطويلة وتفاصيلها القانونية.",
        "text_en": "Paragraph one.\n\nParagraph two.",
        "is_repealed": False,
        "source_page": 100,
    }
    # Force small max chars to trigger splitting
    chunker = ArticleChunker(max_chars_per_chunk=30, split_paragraphs=True)
    chunks = chunker.chunk_article(long_article)
    assert len(chunks) == 2
    assert chunks[0].citation == "Egyptian Civil Code, Article 999"
    assert chunks[1].citation == "Egyptian Civil Code, Article 999"
    assert chunks[0].paragraph_index == 0
    assert chunks[1].paragraph_index == 1
