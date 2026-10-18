"""LLM generator supporting mock, local, and OpenAI-compatible API providers."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator

import httpx

from legal_rag.config import settings

logger = logging.getLogger("legal_rag.rag.generator")


class LegalGenerator:
    """Handles text generation via configured LLM provider."""

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.model_name = model_name or settings.LLM_MODEL_NAME
        self.api_base = api_base or settings.LLM_API_BASE
        self.api_key = api_key or settings.LLM_API_KEY
        logger.info(
            f"Initialized LegalGenerator with provider='{self.provider}', model='{self.model_name}'"
        )

    async def generate_response(self, prompt: str, system_prompt: str) -> str:
        """Generate response text asynchronously."""
        if self.provider == "mock":
            return self._generate_mock(prompt)
        elif self.provider in ("openai_compatible", "vllm"):
            return await self._generate_openai_compatible(prompt, system_prompt)
        else:
            return self._generate_mock(prompt)

    async def generate_stream(self, prompt: str, system_prompt: str) -> AsyncGenerator[str, None]:
        """Yield streamed tokens asynchronously."""
        if self.provider == "mock":
            full_response = self._generate_mock(prompt)
            words = full_response.split(" ")
            for w in words:
                yield w + " "
        elif self.provider in ("openai_compatible", "vllm"):
            url = f"{self.api_base or 'http://localhost:8000/v1'}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key or 'EMPTY'}"}
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "temperature": settings.LLM_TEMPERATURE,
            }
            async with (
                httpx.AsyncClient(timeout=30.0) as client,
                client.stream("POST", url, headers=headers, json=payload) as response,
            ):
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        data = json.loads(line[6:])
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta

    def _generate_mock(self, prompt: str) -> str:
        """Produce a deterministic grounded response based on context for tests and CI."""
        citations = re.findall(r"Egyptian Civil Code, Article \d+", prompt)
        repealed = "مادة ملغاة" in prompt or "repealed" in prompt.lower()

        if repealed:
            cited = citations[0] if citations else "المادة المطلوبة"
            return (
                f"وفقاً لأحكام القانون المدني المصري، فإن {cited} ملغاة بموجب القانون ولا تسري أحكامها حالياً.\n\n"
                "تنويه: هذه معلومات قانونية استرشادية لأغراض تعليمية ولا تُعد بديلاً عن الاستشارة القانونية المهنية."
            )
        elif citations:
            cited_list = ", ".join(dict.fromkeys(citations))
            return (
                f"وفقاً لما ورد في {cited_list} من القانون المدني المصري:\n"
                "تنص الأحكام القانونية ذات الصلة على تنظيم الحقوق والالتزامات المترتبة بين المتعاقدين أو الأطراف المعنية طبقاً لما يقضي به القانون والاتفاق الساري.\n\n"
                "تنويه: هذه معلومات قانونية استرشادية لأغراض تعليمية ولا تُعد بديلاً عن الاستشارة القانونية المهنية."
            )
        else:
            return (
                "لا يتضمن القانون المدني المصري في النصوص المتوفرة دليلاً كافياً للإجابة عن هذا السؤال.\n\n"
                "تنويه: هذه معلومات قانونية استرشادية لأغراض تعليمية ولا تُعد بديلاً عن الاستشارة القانونية المهنية."
            )

    async def _generate_openai_compatible(self, prompt: str, system_prompt: str) -> str:
        """Call remote OpenAI-compatible / vLLM endpoint."""
        url = f"{self.api_base or 'http://localhost:8000/v1'}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key or 'EMPTY'}"}
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": settings.LLM_TEMPERATURE,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(url, headers=headers, json=payload)
            res.raise_for_status()
            data = res.json()
            return data["choices"][0]["message"]["content"]
