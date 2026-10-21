"""Input guardrails for Legal RAG: PII Redaction and Prompt Injection Detection.

Instruments decisions via legal_rag.monitoring.metrics.record_guardrail.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from typing import Any

from legal_rag.monitoring.metrics import record_guardrail

logger = logging.getLogger("legal_rag.guardrails.input")

# Regular expressions for Egyptian PII formats
EGYPTIAN_NATIONAL_ID_REGEX = re.compile(r"\b[23]\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{7}\b")
EGYPTIAN_PHONE_REGEX = re.compile(r"\b(?:\+?20|0)?1[0125]\d{8}\b")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")

# Known direct prompt injection patterns (Arabic and English)
INJECTION_KEYWORDS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard previous",
    "system prompt",
    "developer mode",
    "jailbreak",
    "bypass restrictions",
    "تجاهل التعليمات السابقة",
    "تجاهل كل الأوامر",
    "انسى ما سبق",
    "تصرف كأنك مخترق",
    "أظهر لي موجه النظام",
]


@dataclass
class GuardrailResult:
    """Outcome of a guardrail evaluation."""

    passed: bool
    sanitized_text: str
    flagged: bool
    reason: str | None = None
    metadata: dict[str, Any] | None = None


def redact_pii(text: str) -> GuardrailResult:
    """Detect and redact Egyptian National IDs, phone numbers, and email addresses.

    Replaces PII with sanitized placeholder tokens while preserving legal query semantics.
    """
    flagged = False
    redacted = text

    # 1. National ID (14 digits)
    if EGYPTIAN_NATIONAL_ID_REGEX.search(redacted):
        redacted = EGYPTIAN_NATIONAL_ID_REGEX.sub("[REDACTED_NATIONAL_ID]", redacted)
        flagged = True

    # 2. Egyptian Phone numbers
    if EGYPTIAN_PHONE_REGEX.search(redacted):
        redacted = EGYPTIAN_PHONE_REGEX.sub("[REDACTED_PHONE]", redacted)
        flagged = True

    # 3. Emails
    if EMAIL_REGEX.search(redacted):
        redacted = EMAIL_REGEX.sub("[REDACTED_EMAIL]", redacted)
        flagged = True

    decision = "redacted" if flagged else "clean"
    record_guardrail("pii_redaction", decision)

    return GuardrailResult(
        passed=True,  # Non-blocking: sanitizes inline and allows query to proceed
        sanitized_text=redacted,
        flagged=flagged,
        reason="PII detected and redacted" if flagged else None,
    )


def detect_prompt_injection(text: str) -> GuardrailResult:
    """Detect adversarial prompt injection attempts across direct, encoded, and disguised payloads."""
    lowered = text.lower()

    # 1. Direct keyword override detection
    for kw in INJECTION_KEYWORDS:
        if kw in lowered:
            logger.warning(f"Prompt injection detected (direct keyword: '{kw}')")
            record_guardrail("prompt_injection", "blocked")
            return GuardrailResult(
                passed=False,
                sanitized_text="",
                flagged=True,
                reason=f"Adversarial instruction detected: '{kw}'",
            )

    # 2. Base64 encoded payload detection
    base64_candidates = re.findall(r"[A-Za-z0-9+/]{20,}={0,2}", text)
    for cand in base64_candidates:
        try:
            decoded = base64.b64decode(cand).decode("utf-8", errors="ignore").lower()
            if any(kw in decoded for kw in INJECTION_KEYWORDS):
                logger.warning(f"Prompt injection detected in base64 payload: '{cand}'")
                record_guardrail("prompt_injection", "blocked")
                return GuardrailResult(
                    passed=False,
                    sanitized_text="",
                    flagged=True,
                    reason="Encoded prompt injection payload detected",
                )
        except Exception:  # noqa: S112, BLE001
            continue

    # 3. Disguised statutory instruction (e.g. fake article overriding rules)
    disguised_pattern = re.compile(
        r"(?:مادة|article)\s*\d+.*?(?:تجاهل|ignore|bypass|system)", re.IGNORECASE
    )
    if disguised_pattern.search(text):
        logger.warning("Prompt injection detected in fake statutory provision")
        record_guardrail("prompt_injection", "blocked")
        return GuardrailResult(
            passed=False,
            sanitized_text="",
            flagged=True,
            reason="Adversarial context manipulation detected",
        )

    record_guardrail("prompt_injection", "clean")
    return GuardrailResult(passed=True, sanitized_text=text, flagged=False)
