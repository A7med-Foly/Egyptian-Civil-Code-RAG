"""Arabic and multilingual text normalization for legal documents."""

from __future__ import annotations

import re

# Translation table for Arabic-Indic numerals to Western digits
AR_INDIC_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
WESTERN_TO_AR_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

# Arabic unicode range
ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF]")
DIACRITICS_PATTERN = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # Tashkeel + Tatweel


def is_arabic_char(char: str) -> bool:
    """Check if character is within Arabic Unicode block."""
    return bool(ARABIC_CHAR_PATTERN.match(char))


def is_arabic_word(word: str) -> bool:
    """Check if word contains at least one Arabic character."""
    return any(is_arabic_char(c) for c in word)


def fix_reversed_word(word: str) -> str:
    """Reverse characters in an Arabic word extracted backwards from PDF stream."""
    if is_arabic_word(word):
        return word[::-1]
    return word


def normalize_arabic_indic_number(num_str: str, was_reversed_in_pdf: bool = True) -> int:
    """Convert an Arabic-Indic numeral string into a Western integer."""
    clean_digits = re.sub(r"[^\d٠-٩]", "", num_str)
    if not clean_digits:
        raise ValueError(f"No digits found in: {num_str}")
    if was_reversed_in_pdf:
        clean_digits = clean_digits[::-1]
    western_str = clean_digits.translate(AR_INDIC_TO_WESTERN)
    return int(western_str)


def strip_tashkeel(text: str) -> str:
    """Remove Arabic diacritics (harakat/tashkeel) and tatweel."""
    return DIACRITICS_PATTERN.sub("", text)


def normalize_arabic_orthography(text: str) -> str:
    """Normalize Arabic orthographic variants (alef forms, hamzas, etc.)."""
    # Normalize Alef forms (أ, إ, آ -> ا)
    text = re.sub(r"[إأآٱ]", "ا", text)
    # Normalize Alef Maqsura to Yaa (ى -> ي)
    text = re.sub(r"ى\b", "ي", text)
    # Strip tashkeel and tatweel
    text = strip_tashkeel(text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_arabic_stray_latin(text: str) -> str:
    """Remove stray isolated Latin words that bled from adjacent column."""
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        words = line.split()
        good_words = [
            w for w in words if is_arabic_word(w) or w.isdigit() or re.match(r"^[^\w\s]+$", w)
        ]
        if good_words:
            cleaned_lines.append(" ".join(good_words))
    return "\n".join(cleaned_lines)


def clean_legal_text(text: str, is_arabic: bool = True) -> str:
    """Clean and standardize legal text for chunking and search."""
    if not text:
        return ""
    if is_arabic:
        text = strip_tashkeel(text)
        text = filter_arabic_stray_latin(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    else:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text
