"""Integration and validation tests for the Egyptian Civil Code corpus."""

import json
from pathlib import Path

import pytest

from legal_rag.ingestion.validator import CorpusValidator

PROCESSED_JSON = Path("data/processed/egyptian_civil_code.json")


@pytest.fixture(scope="session")
def corpus():
    assert PROCESSED_JSON.exists(), f"Corpus file missing at {PROCESSED_JSON}"
    with open(PROCESSED_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def test_corpus_count_and_continuity(corpus):
    assert len(corpus) == 1149
    article_numbers = [a["article_number"] for a in corpus]
    expected_numbers = list(range(1, 1150))
    assert (
        article_numbers == expected_numbers
    ), "Article numbers must be strictly contiguous from 1 to 1149"


def test_corpus_validator_passes(corpus):
    validator = CorpusValidator(expected_total=1149)
    summary = validator.validate(corpus)
    assert summary["passed"] is True
    assert summary["missing_count"] == 0
    assert summary["empty_ar_count"] == 0


def test_repealed_articles_flagged(corpus):
    articles_by_num = {a["article_number"]: a for a in corpus}
    # Test decree 384 of 1956 repeal (Articles 54 to 80)
    for num in range(54, 81):
        art = articles_by_num[num]
        assert art["is_repealed"] is True
        assert "ألغيت" in art["text_ar"]
        assert art["citation"] == f"Egyptian Civil Code, Article {num}"

    # Test evidence law repeal (Articles 389 to 417)
    for num in range(389, 418):
        art = articles_by_num[num]
        assert art["is_repealed"] is True
        assert "ألغيت" in art["text_ar"]


def test_spot_check_core_articles(corpus):
    articles_by_num = {a["article_number"]: a for a in corpus}

    # Article 1: Sources of law
    art1 = articles_by_num[1]
    assert "تسرى النصوص التشريعية" in art1["text_ar"]
    assert "Sharia" in art1["text_en"] or "provisions" in art1["text_en"]

    # Article 147: Pacta sunt servanda (العقد شريعة المتعاقدين)
    art147 = articles_by_num[147]
    assert "العقد شريعة المتعاقدين" in art147["text_ar"]
    assert "contract makes the law of the parties" in art147["text_en"]
    assert art147["citation"] == "Egyptian Civil Code, Article 147"
    assert art147["is_repealed"] is False

    # Article 1149: Last article
    art1149 = articles_by_num[1149]
    assert "للشركاء الذين اقتسموا" in art1149["text_ar"]
    assert art1149["citation"] == "Egyptian Civil Code, Article 1149"


def test_extractor_missing_file_raises():
    from legal_rag.ingestion.extractor import EgyptianCivilCodeExtractor

    with pytest.raises(FileNotFoundError):
        EgyptianCivilCodeExtractor("non_existent_file.pdf")


def test_extractor_page_column_extraction():
    import pdfplumber

    from legal_rag.ingestion.extractor import EgyptianCivilCodeExtractor

    pdf_path = Path("data/raw/Egyptian-Civil-Code.pdf")
    if not pdf_path.exists():
        pytest.skip("Raw PDF not found")
    extractor = EgyptianCivilCodeExtractor(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[1]
        ar_lines, en_lines = extractor._extract_page_columns(page)
        assert len(ar_lines) > 0
        assert len(en_lines) > 0
        # Assert Arabic line contains Arabic characters
        assert any("مادة" in l for _, l in ar_lines)
        assert any("Article" in l for _, l in en_lines)


def test_corpus_validator_detects_errors():
    from legal_rag.ingestion.validator import CorpusValidationError

    # 1. Duplicate detection
    v = CorpusValidator(expected_total=3)
    with pytest.raises(CorpusValidationError, match="Found duplicate"):
        v.validate(
            [
                {
                    "article_number": 1,
                    "text_ar": "نص 1",
                    "citation": "Egyptian Civil Code, Article 1",
                },
                {
                    "article_number": 1,
                    "text_ar": "نص 1 مكرر",
                    "citation": "Egyptian Civil Code, Article 1",
                },
                {
                    "article_number": 2,
                    "text_ar": "نص 2",
                    "citation": "Egyptian Civil Code, Article 2",
                },
            ]
        )

    # 2. Missing detection
    with pytest.raises(CorpusValidationError, match="Missing expected"):
        v.validate(
            [
                {
                    "article_number": 1,
                    "text_ar": "نص 1",
                    "citation": "Egyptian Civil Code, Article 1",
                },
                {
                    "article_number": 3,
                    "text_ar": "نص 3",
                    "citation": "Egyptian Civil Code, Article 3",
                },
            ]
        )

    # 3. Empty text and citation mismatch
    v1 = CorpusValidator(expected_total=1)
    with pytest.raises(CorpusValidationError, match="empty text_ar"):
        v1.validate(
            [{"article_number": 1, "text_ar": "", "citation": "bad citation", "is_repealed": False}]
        )

    # 4. Giant record detection
    v_giant = CorpusValidator(expected_total=1, max_char_length=20)
    with pytest.raises(CorpusValidationError, match="Giant records"):
        v_giant.validate(
            [
                {
                    "article_number": 1,
                    "text_ar": "نص طويل يتجاوز الحد المسموح به",
                    "citation": "Egyptian Civil Code, Article 1",
                    "is_repealed": False,
                }
            ]
        )


def test_validate_file_helper(tmp_path):
    from legal_rag.ingestion.validator import CorpusValidationError, validate_file

    dummy = tmp_path / "valid.json"
    dummy.write_text("[]", encoding="utf-8")
    with pytest.raises(CorpusValidationError):
        validate_file(dummy)
