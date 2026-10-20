"""Redis Streams event producer for legal queries."""

from __future__ import annotations

import argparse
import logging
import os
import time
import uuid

import redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("legal_rag.serving.stream_producer")


class LegalStreamProducer:
    """Publishes statutory legal queries to Redis stream."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        stream_key: str = "legal_rag_requests",
    ):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.stream_key = stream_key
        self.client = redis.Redis(host=self.host, port=self.port, decode_responses=True)

    def publish_query(self, question: str, top_k: int = 3) -> str:
        """Publish a single query event to the stream with microsecond timestamp."""
        query_id = str(uuid.uuid4())
        event_payload = {
            "query_id": query_id,
            "question": question,
            "top_k": str(top_k),
            "timestamp_ms": str(time.time() * 1000),
        }
        msg_id = self.client.xadd(self.stream_key, event_payload)
        logger.info(
            f"Published event {msg_id} (query_id={query_id}) to stream '{self.stream_key}'."
        )
        return msg_id


def main():
    parser = argparse.ArgumentParser(description="Publish legal queries to Redis Stream.")
    parser.add_argument("--count", type=int, default=5, help="Number of events to emit")
    parser.add_argument("--host", type=str, default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    args = parser.parse_args()

    questions = [
        "ما هي شروط صحة عقد البيع وفق القانون المدني المصري؟",
        "ما حكم الالتزام الطبيعي في القانون المدني؟",
        "ما هي حالات فسخ العقد لعدم التنفيذ؟",
        "ما هي المسؤولية التقصيرية عن أعمال التابع؟",
        "ما هو الوضع القانوني للمادة 60 المتعلقة بالأحوال المدنية؟",
    ]

    producer = LegalStreamProducer(host=args.host, port=args.port)
    for i in range(args.count):
        q = questions[i % len(questions)]
        producer.publish_query(q)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
