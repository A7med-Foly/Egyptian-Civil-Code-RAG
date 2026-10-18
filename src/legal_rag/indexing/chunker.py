"""Article-aware chunking strategies for Arabic legal documents."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


class LegalChunk(BaseModel):
    """Represents an indexable chunk of a legal document."""

    chunk_id: str
    article_number: int
    citation: str
    book: str
    chapter: str
    section: str
    topic: str
    is_repealed: bool
    text: str
    text_en: str
    source_page: int
    paragraph_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArticleChunker:
    """Article-level chunking that preserves citation integrity and statutory structure."""

    def __init__(self, max_chars_per_chunk: int = 2000, split_paragraphs: bool = True):
        self.max_chars_per_chunk = max_chars_per_chunk
        self.split_paragraphs = split_paragraphs
        # Regex to split Arabic numbered paragraphs e.g. (١( or )١( or 1-
        self.para_split_pat = re.compile(r"(?=(?:[\(\[\{]?[٠-٩\d]+[\)\]\}]?\s*))")

    def chunk_article(self, article: dict[str, Any]) -> list[LegalChunk]:
        """Convert a single article dictionary into one or more LegalChunks."""
        art_num = article["article_number"]
        citation = article.get("citation", f"Egyptian Civil Code, Article {art_num}")
        text_ar = (article.get("text_ar") or "").strip()
        text_en = (article.get("text_en") or "").strip()
        is_repealed = article.get("is_repealed", False)
        source_page = article.get("source_page", 1)

        base_meta = {
            "book": article.get("book", ""),
            "chapter": article.get("chapter", ""),
            "section": article.get("section", ""),
            "topic": article.get("topic", ""),
        }

        # If article fits within max_chars or paragraph splitting disabled, return single chunk
        if len(text_ar) <= self.max_chars_per_chunk or not self.split_paragraphs:
            return [
                LegalChunk(
                    chunk_id=f"art_{art_num}_0",
                    article_number=art_num,
                    citation=citation,
                    book=base_meta["book"],
                    chapter=base_meta["chapter"],
                    section=base_meta["section"],
                    topic=base_meta["topic"],
                    is_repealed=is_repealed,
                    text=text_ar,
                    text_en=text_en,
                    source_page=source_page,
                    paragraph_index=0,
                    metadata=base_meta,
                )
            ]

        # For long articles, split by numbered paragraphs
        paragraphs = [p.strip() for p in text_ar.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text_ar.split("\n") if p.strip()]

        chunks: list[LegalChunk] = []
        for idx, para in enumerate(paragraphs):
            chunks.append(
                LegalChunk(
                    chunk_id=f"art_{art_num}_{idx}",
                    article_number=art_num,
                    citation=citation,
                    book=base_meta["book"],
                    chapter=base_meta["chapter"],
                    section=base_meta["section"],
                    topic=base_meta["topic"],
                    is_repealed=is_repealed,
                    text=para,
                    text_en=text_en if idx == 0 else "",
                    source_page=source_page,
                    paragraph_index=idx,
                    metadata=base_meta,
                )
            )

        return (
            chunks
            if chunks
            else [
                LegalChunk(
                    chunk_id=f"art_{art_num}_0",
                    article_number=art_num,
                    citation=citation,
                    book=base_meta["book"],
                    chapter=base_meta["chapter"],
                    section=base_meta["section"],
                    topic=base_meta["topic"],
                    is_repealed=is_repealed,
                    text=text_ar,
                    text_en=text_en,
                    source_page=source_page,
                    paragraph_index=0,
                    metadata=base_meta,
                )
            ]
        )

    def chunk_corpus(self, articles: list[dict[str, Any]]) -> list[LegalChunk]:
        """Convert the full corpus into a list of LegalChunks."""
        all_chunks: list[LegalChunk] = []
        for art in articles:
            all_chunks.extend(self.chunk_article(art))
        return all_chunks
