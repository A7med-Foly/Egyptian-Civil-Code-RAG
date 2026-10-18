"""Unit tests for Arabic and legal text normalizer."""

import pytest

from legal_rag.ingestion.normalizer import (
    clean_legal_text,
    filter_arabic_stray_latin,
    fix_reversed_word,
    is_arabic_char,
    is_arabic_word,
    normalize_arabic_indic_number,
    normalize_arabic_orthography,
    strip_tashkeel,
)


def test_is_arabic_char():
    assert is_arabic_char("ع") is True
    assert is_arabic_char("a") is False
    assert is_arabic_char("١") is True


def test_is_arabic_word():
    assert is_arabic_word("القانون") is True
    assert is_arabic_word("Contract") is False
    assert is_arabic_word("147") is False


def test_fix_reversed_word():
    assert fix_reversed_word("يرصملا") == "المصري"
    assert fix_reversed_word("Law") == "Law"


@pytest.mark.parametrize(
    "raw_str,was_reversed,expected",
    [
        ("٧", True, 7),
        ("٨", False, 8),
        ("٥٠١", True, 105),
        ("٩٤١١", True, 1149),
        ("147", False, 147),
    ],
)
def test_normalize_arabic_indic_number(raw_str, was_reversed, expected):
    assert normalize_arabic_indic_number(raw_str, was_reversed_in_pdf=was_reversed) == expected


def test_strip_tashkeel():
    vocalized = "العَقْدُ شَرِيعَةُ المُتَعَاقِدِينَ"
    plain = "العقد شريعة المتعاقدين"
    assert strip_tashkeel(vocalized) == plain


def test_normalize_arabic_orthography():
    variant = "إلغاء أوامر الإثبات فى الدعوى"
    normalized = normalize_arabic_orthography(variant)
    assert "الغاء" in normalized
    assert "اوامر" in normalized
    assert "اثبات" in normalized


def test_filter_arabic_stray_latin():
    mixed = "العقد شريعة المتعاقدين ties ولا يجوز نقضه oss,"
    cleaned = filter_arabic_stray_latin(mixed)
    assert "ties" not in cleaned
    assert "oss," not in cleaned
    assert "العقد شريعة المتعاقدين ولا يجوز نقضه" == cleaned


def test_clean_legal_text():
    raw = "  المادة  147 \n\n\n العقد   شريعة  "
    cleaned = clean_legal_text(raw, is_arabic=True)
    assert "المادة 147" in cleaned
    assert "العقد شريعة" in cleaned
