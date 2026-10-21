"""Output guardrails for Legal RAG: Schema Validation and Citation Grounding."""

from __future__ import annotations

import logging
import re
from typing import Any

from legal_rag.monitoring.metrics import record_guardrail

logger = logging.getLogger("legal_rag.guardrails.output")


def validate_citation_grounding(
    answer: str,
    sources: list[str],
    retrieved_texts: list[str],
) -> dict[str, Any]:
    """Verify that statutory citations cited in the generated answer exist in retrieved context.

    Prevents hallucinated legal citations from reaching legal users.
    """
    # Extract article numbers cited in text (e.g. "المادة 147" or "Article 147")
    cited_articles = set(re.findall(r"(?:المادة|مادة|article)\s*(\d+)", answer, re.IGNORECASE))

    # Extract article numbers present in retrieved context / sources
    grounded_articles = set()
    for s in sources:
        m = re.search(r"(\d+)", s)
        if m:
            grounded_articles.add(m.group(1))

    for txt in retrieved_texts:
        for m in re.findall(r"(?:المادة|مادة|article)\s*(\d+)", txt, re.IGNORECASE):
            grounded_articles.add(m)

    unsupported = cited_articles - grounded_articles
    is_grounded = len(unsupported) == 0

    decision = "grounded" if is_grounded else "unsupported_citation"
    record_guardrail("citation_grounding", decision)

    if not is_grounded:
        logger.warning(
            f"Output guardrail flagged unsupported citations: {unsupported} "
            f"(Grounded context: {grounded_articles})"
        )

    return {
        "is_grounded": is_grounded,
        "cited_articles": sorted(cited_articles),
        "grounded_articles": sorted(grounded_articles),
        "unsupported_citations": sorted(unsupported),
    }
