"""Corpus validation for Egyptian Civil Code structured JSON."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("legal_rag.ingestion.validator")


class CorpusValidationError(Exception):
    """Raised when structured legal corpus fails validation checks."""


class CorpusValidator:
    """Validates the extracted legal articles against official criteria."""

    def __init__(self, expected_total: int = 1149, max_char_length: int = 15000):
        self.expected_total = expected_total
        self.max_char_length = max_char_length

    def validate(self, articles: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate list of article dictionaries and return diagnostic metrics."""
        issues: list[str] = []
        article_numbers = [a.get("article_number") for a in articles]

        # 1. Check for duplicates
        seen = set()
        duplicates = [x for x in article_numbers if x in seen or seen.add(x)]
        if duplicates:
            issues.append(f"Found duplicate article numbers: {duplicates[:10]}")

        # 2. Check for missing or unexpected articles
        art_set = set(article_numbers)
        expected_set = set(range(1, self.expected_total + 1))
        missing = sorted(expected_set - art_set)
        if missing:
            issues.append(
                f"Missing expected article numbers ({len(missing)} total): {missing[:15]}"
            )

        # 3. Check article fields and content
        repealed_count = 0
        active_count = 0
        empty_ar_count = 0
        giant_records: list[int] = []

        for a in articles:
            art_num = a.get("article_number")
            is_repealed = a.get("is_repealed", False)
            text_ar = (a.get("text_ar") or "").strip()
            citation = a.get("citation") or ""

            if is_repealed:
                repealed_count += 1
            else:
                active_count += 1

            if not text_ar:
                empty_ar_count += 1
                issues.append(f"Article {art_num} has empty text_ar")

            if len(text_ar) > self.max_char_length:
                giant_records.append(art_num)

            expected_citation = f"Egyptian Civil Code, Article {art_num}"
            if citation != expected_citation:
                issues.append(
                    f"Article {art_num} citation mismatch: '{citation}' != '{expected_citation}'"
                )

        if giant_records:
            issues.append(
                f"Giant records detected (> {self.max_char_length} chars, possible failed split): {giant_records}"
            )

        # 4. Assert repealed ranges are covered
        expected_repealed_54_80 = set(range(54, 81))
        expected_repealed_389_417 = set(range(389, 418))
        repealed_in_data = {a["article_number"] for a in articles if a.get("is_repealed")}

        if not expected_repealed_54_80.issubset(repealed_in_data):
            issues.append("Not all articles in range 54-80 are flagged as is_repealed=True")
        if not expected_repealed_389_417.issubset(repealed_in_data):
            issues.append("Not all articles in range 389-417 are flagged as is_repealed=True")

        summary = {
            "total_articles": len(articles),
            "expected_articles": self.expected_total,
            "active_articles": active_count,
            "repealed_articles": repealed_count,
            "empty_ar_count": empty_ar_count,
            "missing_count": len(missing),
            "issues": issues,
            "passed": len(issues) == 0,
        }

        if issues:
            logger.error(
                f"Corpus validation FAILED with {len(issues)} issues:\n" + "\n".join(issues[:10])
            )
            raise CorpusValidationError(f"Validation failed: {issues[:5]}")

        logger.info(
            f"Corpus validation PASSED: {len(articles)} articles validated "
            f"({active_count} active, {repealed_count} repealed)."
        )
        return summary


def validate_file(file_path: str | Path) -> dict[str, Any]:
    """Validate a JSON corpus file."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    validator = CorpusValidator()
    return validator.validate(data)


def main():
    """CLI entry point to validate corpus and write validation report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    corpus_path = Path("data/processed/egyptian_civil_code.json")
    report_path = Path("data/processed/validation_report.json")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file {corpus_path} not found.")

    summary = validate_file(corpus_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Validation report saved to {report_path}")


if __name__ == "__main__":
    main()
