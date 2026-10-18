"""ChromaDB vector store with hybrid dense + BM25 sparse search."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from legal_rag.config import settings
from legal_rag.indexing.chunker import ArticleChunker, LegalChunk
from legal_rag.indexing.embedder import LegalEmbedder

logger = logging.getLogger("legal_rag.indexing.vector_store")


class LegalVectorStore:
    """Manages persistent ChromaDB vector storage and BM25 hybrid search."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = "egyptian_civil_code",
        embedder: LegalEmbedder | None = None,
    ):
        self.persist_dir = Path(persist_dir or settings.INDEX_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedder = embedder or LegalEmbedder()

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Egyptian Civil Code Articles"},
        )

        self.bm25: BM25Okapi | None = None
        self.bm25_chunks: list[LegalChunk] = []
        self._init_bm25_from_chroma()

    def _init_bm25_from_chroma(self) -> None:
        """Populate in-memory BM25 index from existing Chroma collection items."""
        count = self.collection.count()
        if count == 0:
            return

        # Fetch documents and metadata from Chroma
        all_data = self.collection.get(include=["documents", "metadatas"])
        docs = all_data.get("documents", [])
        metas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])

        chunks = []
        for doc_id, doc, meta in zip(ids, docs, metas):
            chunk = LegalChunk(
                chunk_id=doc_id,
                article_number=meta.get("article_number", 0),
                citation=meta.get("citation", ""),
                book=meta.get("book", ""),
                chapter=meta.get("chapter", ""),
                section=meta.get("section", ""),
                topic=meta.get("topic", ""),
                is_repealed=meta.get("is_repealed", False),
                text=doc,
                text_en=meta.get("text_en", ""),
                source_page=meta.get("source_page", 1),
                paragraph_index=meta.get("paragraph_index", 0),
            )
            chunks.append(chunk)

        self.bm25_chunks = chunks
        tokenized_corpus = [c.text.split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Initialized BM25 index with {len(chunks)} chunks.")

    @property
    def documents_indexed(self) -> int:
        """Return count of documents in the vector index."""
        return self.collection.count()

    def index_chunks(self, chunks: list[LegalChunk], batch_size: int = 64) -> int:
        """Embed and insert LegalChunks into ChromaDB collection."""
        if not chunks:
            return 0

        logger.info(f"Indexing {len(chunks)} chunks in batches of {batch_size}...")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [
                {
                    "article_number": c.article_number,
                    "citation": c.citation,
                    "book": c.book,
                    "chapter": c.chapter,
                    "section": c.section,
                    "topic": c.topic,
                    "is_repealed": c.is_repealed,
                    "source_page": c.source_page,
                    "paragraph_index": c.paragraph_index,
                }
                for c in batch
            ]
            embeddings = self.embedder.embed_texts(texts).tolist()

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        self._init_bm25_from_chroma()
        return len(chunks)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        alpha: float = 0.6,
    ) -> list[tuple[LegalChunk, float]]:
        """Perform hybrid search combining dense semantic vectors and sparse BM25 scores.

        alpha: weight for dense vector search (1 - alpha for BM25).
        """
        if self.documents_indexed == 0:
            return []

        # 1. Dense retrieval
        query_vec = self.embedder.embed_query(query)
        dense_results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k * 3, self.documents_indexed),
            include=["documents", "metadatas", "distances"],
        )

        dense_scores: dict[str, float] = {}
        dense_chunks: dict[str, LegalChunk] = {}

        if dense_results and dense_results["ids"]:
            ids = dense_results["ids"][0]
            docs = dense_results["documents"][0]
            metas = dense_results["metadatas"][0]
            distances = dense_results["distances"][0]

            for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
                # Cosine similarity roughly 1 - distance/2
                sim = max(0.0, 1.0 - (dist / 2.0))
                dense_scores[doc_id] = sim
                dense_chunks[doc_id] = LegalChunk(
                    chunk_id=doc_id,
                    article_number=meta.get("article_number", 0),
                    citation=meta.get("citation", ""),
                    book=meta.get("book", ""),
                    chapter=meta.get("chapter", ""),
                    section=meta.get("section", ""),
                    topic=meta.get("topic", ""),
                    is_repealed=meta.get("is_repealed", False),
                    text=doc,
                    text_en="",
                    source_page=meta.get("source_page", 1),
                    paragraph_index=meta.get("paragraph_index", 0),
                )

        # 2. Sparse BM25 retrieval
        sparse_scores: dict[str, float] = {}
        if self.bm25 and self.bm25_chunks:
            q_tokens = query.split()
            bm_scores = self.bm25.get_scores(q_tokens)
            max_bm = max(bm_scores) if max(bm_scores) > 0 else 1.0
            for chunk, score in zip(self.bm25_chunks, bm_scores):
                if score > 0:
                    sparse_scores[chunk.chunk_id] = score / max_bm
                    if chunk.chunk_id not in dense_chunks:
                        dense_chunks[chunk.chunk_id] = chunk

        # 3. Reciprocal Rank / Linear Combination Fusion
        candidate_ids = set(dense_scores.keys()).union(set(sparse_scores.keys()))
        combined: list[tuple[LegalChunk, float]] = []

        for cid in candidate_ids:
            d_score = dense_scores.get(cid, 0.0)
            s_score = sparse_scores.get(cid, 0.0)
            final_score = (alpha * d_score) + ((1.0 - alpha) * s_score)
            combined.append((dense_chunks[cid], final_score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]


def main():
    """CLI entry point to index the processed legal corpus into ChromaDB."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    corpus_file = settings.PROCESSED_DATA_PATH
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}. Run ingestion first.")

    with open(corpus_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    chunker = ArticleChunker()
    chunks = chunker.chunk_corpus(articles)
    logger.info(f"Chunked {len(articles)} articles into {len(chunks)} chunks.")

    vstore = LegalVectorStore()
    vstore.index_chunks(chunks)
    logger.info(f"Index complete. Total chunks in ChromaDB: {vstore.documents_indexed}")


if __name__ == "__main__":
    main()
