"""Layout-aware extraction of the bilingual Egyptian Civil Code PDF into structured JSON."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pdfplumber

from legal_rag.ingestion.normalizer import (
    clean_legal_text,
    fix_reversed_word,
    normalize_arabic_indic_number,
)

logger = logging.getLogger("legal_rag.ingestion.extractor")

REPEALED_RANGES = [
    {
        "start": 54,
        "end": 80,
        "reason_ar": "ألغيت المواد من 54 إلى 80 بموجب القرار الجمهوري بالقانون رقم 384 لسنة 1956 (أحكام الولاية على المال).",
        "reason_en": "Articles 54 to 80 were repealed by Presidential Decree Law No. 384 of 1956.",
        "page": 7,
    },
    {
        "start": 389,
        "end": 417,
        "reason_ar": "ألغيت المواد من 389 إلى 417 بموجب قانون الإثبات رقم 25 لسنة 1968.",
        "reason_en": "Articles 389 to 417 were repealed by the Law of Evidence No. 25 of 1968.",
        "page": 53,
    },
]


class EgyptianCivilCodeExtractor:
    """Extracts articles and legal hierarchy from the bilingual Egyptian Civil Code PDF."""

    def __init__(self, pdf_path: str | Path, column_split_x: float = 285.0):
        self.pdf_path = Path(pdf_path)
        self.column_split_x = column_split_x
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

    def _extract_page_columns(
        self, page: pdfplumber.page.Page
    ) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
        """Extract sorted lines from Arabic (right) and English (left) columns."""
        width = float(page.width)
        height = float(page.height)

        # 1. Arabic side (right)
        ar_crop = page.crop((self.column_split_x - 10, 0, width, height))
        ar_words = ar_crop.extract_words()
        ar_lines_dict: dict[int, list[dict[str, Any]]] = {}
        for w in ar_words:
            top_bucket = round(w["top"] / 4.0) * 4
            ar_lines_dict.setdefault(top_bucket, []).append(w)

        ar_lines: list[tuple[float, str]] = []
        for top in sorted(ar_lines_dict.keys()):
            lw = sorted(ar_lines_dict[top], key=lambda x: x["x0"], reverse=True)
            fixed = " ".join(fix_reversed_word(w["text"]) for w in lw).strip()
            if fixed:
                ar_lines.append((float(top), fixed))

        # 2. English side (left)
        en_crop = page.crop((0, 0, self.column_split_x + 10, height))
        en_words = en_crop.extract_words()
        en_lines_dict: dict[int, list[dict[str, Any]]] = {}
        for w in en_words:
            top_bucket = round(w["top"] / 4.0) * 4
            en_lines_dict.setdefault(top_bucket, []).append(w)

        en_lines: list[tuple[float, str]] = []
        for top in sorted(en_lines_dict.keys()):
            lw = sorted(en_lines_dict[top], key=lambda x: x["x0"])
            line_str = " ".join(w["text"] for w in lw).strip()
            if line_str:
                en_lines.append((float(top), line_str))

        return ar_lines, en_lines

    def extract_all(self) -> list[dict[str, Any]]:
        """Process all pages and return a list of structured article dictionaries."""
        logger.info(f"Opening PDF: {self.pdf_path}")
        articles: dict[int, dict[str, Any]] = {}

        # Populate repealed articles upfront with official annotations
        for r in REPEALED_RANGES:
            for art_num in range(r["start"], r["end"] + 1):
                articles[art_num] = {
                    "article_number": art_num,
                    "book": (
                        "Obligations or Personal Rights" if art_num <= 417 else "General Provisions"
                    ),
                    "chapter": "Repealed Section",
                    "section": "",
                    "topic": "Repealed by Law",
                    "text_ar": r["reason_ar"],
                    "text_en": r["reason_en"],
                    "is_repealed": True,
                    "source_page": r["page"],
                    "citation": f"Egyptian Civil Code, Article {art_num}",
                }

        # Hierarchy tracking
        current_book = "General Provisions / Preliminary Part"
        current_chapter = "General Provisions"
        current_section = ""
        current_topic = ""

        # Strict regex patterns matching article HEADERS (anchored at line start)
        ar_header_pat = re.compile(r"^\s*مادة\s*\(?\s*([٠-٩\d]+)\s*\)?")
        en_header_pat = re.compile(r"^\s*Article\s*(\d+)\b", re.IGNORECASE)

        current_art_num: int | None = None
        current_ar_text_parts: list[str] = []
        current_en_text_parts: list[str] = []
        current_page_num = 1

        def flush_current():
            nonlocal current_art_num, current_ar_text_parts, current_en_text_parts
            if current_art_num is not None:
                ar_body = clean_legal_text("\n".join(current_ar_text_parts), is_arabic=True)
                en_body = clean_legal_text("\n".join(current_en_text_parts), is_arabic=False)
                if current_art_num in articles:
                    if not articles[current_art_num]["is_repealed"]:
                        if ar_body:
                            existing_ar = articles[current_art_num]["text_ar"]
                            articles[current_art_num]["text_ar"] = clean_legal_text(
                                f"{existing_ar}\n{ar_body}" if existing_ar else ar_body
                            )
                        if en_body:
                            existing_en = articles[current_art_num]["text_en"]
                            articles[current_art_num]["text_en"] = clean_legal_text(
                                f"{existing_en}\n{en_body}" if existing_en else en_body
                            )
                else:
                    articles[current_art_num] = {
                        "article_number": current_art_num,
                        "book": current_book,
                        "chapter": current_chapter,
                        "section": current_section,
                        "topic": current_topic,
                        "text_ar": ar_body,
                        "text_en": en_body,
                        "is_repealed": False,
                        "source_page": current_page_num,
                        "citation": f"Egyptian Civil Code, Article {current_art_num}",
                    }
            current_ar_text_parts = []
            current_en_text_parts = []

        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Processing {total_pages} pages...")

            for p_idx, page in enumerate(pdf.pages):
                current_page_num = p_idx + 1
                ar_lines, en_lines = self._extract_page_columns(page)

                # Find valid article header markers on this page with Y coordinates
                page_ar_markers: list[tuple[float, int, str]] = []
                for top, line in ar_lines:
                    m = ar_header_pat.match(line)
                    if m:
                        raw = m.group(1)
                        try:
                            if any("\u0660" <= ch <= "\u0669" for ch in raw):
                                num = normalize_arabic_indic_number(raw, was_reversed_in_pdf=True)
                            else:
                                num = int(raw)
                            if 1 <= num <= 1149:
                                remainder = line[m.end() :].strip()
                                page_ar_markers.append((top, num, remainder))
                        except (ValueError, KeyError, IndexError):
                            pass

                page_en_markers: list[tuple[float, int, str]] = []
                for top, line in en_lines:
                    m = en_header_pat.match(line)
                    if m:
                        num = int(m.group(1))
                        if 1 <= num <= 1149:
                            remainder = line[m.end() :].strip()
                            page_en_markers.append((top, num, remainder))

                # Combine markers by top coordinate
                # Store (art_num, top, remainder_ar, remainder_en)
                combined_markers: dict[int, float] = {}
                for top, num, rem in page_en_markers:
                    combined_markers[num] = top
                for top, num, rem in page_ar_markers:
                    if num not in combined_markers:
                        combined_markers[num] = top

                sorted_markers = sorted(combined_markers.items(), key=lambda x: x[1])

                # Split lines per marker
                if sorted_markers:
                    first_top = sorted_markers[0][1]
                    for top, line in ar_lines:
                        if top < first_top - 5:
                            current_ar_text_parts.append(line)
                    for top, line in en_lines:
                        if top < first_top - 5:
                            current_en_text_parts.append(line)

                    # Now process each article block on this page
                    for idx, (art_num, m_top) in enumerate(sorted_markers):
                        flush_current()
                        current_art_num = art_num
                        next_top = (
                            sorted_markers[idx + 1][1] - 5
                            if idx + 1 < len(sorted_markers)
                            else float("inf")
                        )

                        for top, line in ar_lines:
                            if m_top - 5 <= top < next_top:
                                # Update hierarchy if heading in line
                                if any(h in line for h in ["الكتاب", "الباب"]):
                                    current_book = line
                                elif "الفصل" in line:
                                    current_chapter = line
                                elif any(h in line for h in ["المبحث", "القسم"]):
                                    current_section = line

                                m = ar_header_pat.match(line)
                                if m:
                                    remainder = line[m.end() :].strip()
                                    if remainder:
                                        current_ar_text_parts.append(remainder)
                                else:
                                    current_ar_text_parts.append(line)

                        for top, line in en_lines:
                            if m_top - 5 <= top < next_top:
                                if any(h in line for h in ["BOOK", "Book"]) or any(
                                    h in line for h in ["Chapter", "CHAPTER"]
                                ):
                                    pass
                                m = en_header_pat.match(line)
                                if m:
                                    remainder = line[m.end() :].strip()
                                    if remainder:
                                        current_en_text_parts.append(remainder)
                                else:
                                    current_en_text_parts.append(line)
                else:
                    # Page has no new markers -> continuation page
                    for top, line in ar_lines:
                        current_ar_text_parts.append(line)
                    for top, line in en_lines:
                        current_en_text_parts.append(line)

            # Flush final article
            flush_current()

        # Handle Article 1022 if missing Arabic body due to omitted header in PDF
        if 1022 in articles and not articles[1022].get("text_ar"):
            articles[1022]["text_ar"] = (
                "نفقات الأعمال اللازمة لاستعمال حق الارتفاق والمحافظة عليه تكون على عاتق صاحب العقار المرتفق، "
                "هذا ما لم يشترط غير ذلك. فإذا كان صاحب العقار المرتفق به هو المكلف بأن يقوم بتلك الأعمال على نفقته، "
                "كان له دائماً أن يتخلص من هذا التكليف بالتخلي عن العقار المرتفق به كله أو بعضه لصاحب العقار المرتفق."
            )

        sorted_articles = [articles[k] for k in sorted(articles.keys())]
        logger.info(f"Extraction complete. Total articles extracted: {len(sorted_articles)}")
        return sorted_articles

    def save_json(self, output_path: str | Path) -> Path:
        """Extract and save structured corpus to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        articles = self.extract_all()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(articles)} articles to {output_path}")
        return output_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pdf_path = Path("data/raw/Egyptian-Civil-Code.pdf")
    output_path = Path("data/processed/egyptian_civil_code.json")
    extractor = EgyptianCivilCodeExtractor(pdf_path)
    extractor.save_json(output_path)


if __name__ == "__main__":
    main()
