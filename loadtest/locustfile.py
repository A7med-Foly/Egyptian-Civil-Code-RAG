"""Locust load test suite for Egyptian Legal RAG.

Workload Distribution:
- 80% Single Question Answering (/ask)
- 15% Batch Inquiries (/ask_batch)
- 5% System Diagnostics (/health, /metadata)
Simulates realistic user think-time with between(1, 3) seconds wait time.
Uses diverse legal queries across different Egyptian Civil Code books to avoid caching bias.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# Diverse legal queries covering contract law, property law, civil responsibility, and prescription
STATUTORY_QUERIES = [
    "ما هي شروط صحة التراضي في القانون المدني المصري؟",
    "ما هو سن الرشد القانوني وما هي عوارض الأهلية؟",
    "ما هي أحكام الغبن والاستغلال وفقاً للمادة 129؟",
    "ما هو حكم بيع ملك الغير وهل يجوز إبطاله؟",
    "ما هي التزامات البائع بشأن ضمان العيوب الخفية؟",
    "متى تسقط دعوى الضمان بالتقادم في عقد البيع؟",
    "ما هي آثار حوالة الحق بالنسبة للمدين والغير؟",
    "كيف تنقضي الالتزامات بالمقاصة القانونية؟",
    "ما هي شروط تحقق المسؤولية التقصيرية عن العمل غير المشروع؟",
    "ما هي مسؤولية متولي الرقابة عن الضرر الذي يحدثه القاصر؟",
    "ما هي أحكام عقد الإيجار وحقوق المستأجر في الانتفاع بالعين؟",
    "ما هي أسباب انقضاء حق الارتفاق وفق القانون المدني؟",
    "ما هو الفرق بين حق الملكية التامة وحق الانتفاع؟",
    "ما هي الشروط الجوهرية لصحة الرهن التأميني وقيده؟",
    "ما هي حقوق الدائن المرتهن في استيفاء حقه بالأولوية؟",
    "ما هي مدة التقادم المسقط للحقوق العامة في القانون المدني؟",
    "ما هو الوضع القانوني للمادة 60 المتعلقة بالأحوال المدنية؟",
    "هل يجوز تعديل شروط العقد باتفاق الطرفين لاحقاً؟",
    "ما هي شروط دعوى عدم نفاذ تصرفات المدين (الدعوى البولصية)؟",
    "ما هو أثر القوة القاهرة أو الحادث الفجائي على الالتزام العقدي؟",
]


class LegalRAGUser(HttpUser):
    """Simulates an active legal researcher or advocate interacting with the Q&A service."""

    wait_time = between(1.0, 3.0)

    @task(16)
    def ask_single_question(self):
        """80% traffic: single question answering with dynamic statutory query."""
        question = random.choice(STATUTORY_QUERIES)
        payload = {"question": question, "top_k": 3}
        headers = {"Content-Type": "application/json"}

        with self.client.post(
            "/ask",
            json=payload,
            headers=headers,
            catch_response=True,
            name="/ask (single)",
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "answer" in data and ("sources" in data or "model_version" in data):
                        response.success()
                    else:
                        response.failure(f"Missing expected response fields: {data.keys()}")
                except Exception as e:  # noqa: BLE001
                    response.failure(f"Invalid JSON response: {e}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text[:100]}")

    @task(3)
    def ask_batch_questions(self):
        """15% traffic: batch query processing."""
        sample_q = random.sample(STATUTORY_QUERIES, k=2)
        payload = {"questions": sample_q}
        headers = {"Content-Type": "application/json"}

        with self.client.post(
            "/ask_batch",
            json=payload,
            headers=headers,
            catch_response=True,
            name="/ask_batch",
        ) as response:
            # Fallback for servers that only support single /ask
            if response.status_code == 404:
                # Fallback to parallel single query for CAT 1 baseline
                single_payload = {"question": sample_q[0], "top_k": 3}
                self.client.post("/ask", json=single_payload, name="/ask (batch-fallback)")
                response.success()
            elif response.status_code == 200:
                response.success()
            else:
                response.failure(f"Batch request failed with HTTP {response.status_code}")

    @task(1)
    def check_health_and_metadata(self):
        """5% traffic: health check and system metadata diagnostics."""
        target = random.choice(["/health", "/metadata"])
        with self.client.get(target, catch_response=True, name=target) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Diagnostic probe {target} returned {response.status_code}")
