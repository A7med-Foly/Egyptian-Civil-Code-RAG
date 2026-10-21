"""Redis token-bucket rate limiter with honest 429 Retry-After response.

Tracks requests per client / API key, prevents server overload, and exposes
throttling metrics to Prometheus.
"""

from __future__ import annotations

import logging
import os
import time
from typing import NamedTuple

from legal_rag.monitoring.metrics import record_rate_limit

logger = logging.getLogger("legal_rag.api.rate_limiter")


class RateLimitResult(NamedTuple):
    allowed: bool
    remaining_tokens: int
    retry_after_seconds: int


class TokenBucketRateLimiter:
    """Token bucket rate limiter backed by Redis with in-memory fallback.

    Refills tokens continuously at refill_rate (tokens/sec) up to capacity.
    """

    def __init__(
        self,
        capacity: int = 20,
        refill_rate: float = 2.0,  # 2 tokens per second
        redis_url: str | None = None,
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = None

        # Try connecting to Redis
        try:
            import redis

            client = redis.from_url(self.redis_url, socket_timeout=0.5)
            client.ping()
            self.redis_client = client
            logger.info(f"Connected to Redis for token bucket rate limiting at {self.redis_url}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Redis unavailable ({e}); using local memory token bucket.")
            self.redis_client = None

        # Local in-memory store: {client_id: (tokens, last_refill_timestamp)}
        self._memory_buckets: dict[str, tuple[float, float]] = {}

    def check_rate_limit(self, client_id: str, tokens_needed: int = 1) -> RateLimitResult:
        """Check if client has sufficient tokens available; consume tokens if allowed."""
        now = time.time()

        if self.redis_client is not None:
            try:
                return self._check_redis(client_id, tokens_needed, now)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"Redis error during rate limit check ({e}); failing over to memory."
                )

        return self._check_memory(client_id, tokens_needed, now)

    def _check_memory(self, client_id: str, tokens_needed: int, now: float) -> RateLimitResult:
        tokens, last_time = self._memory_buckets.get(client_id, (self.capacity, now))

        # Refill tokens proportional to elapsed time
        elapsed = now - last_time
        tokens = min(self.capacity, tokens + elapsed * self.refill_rate)

        if tokens >= tokens_needed:
            tokens -= tokens_needed
            self._memory_buckets[client_id] = (tokens, now)
            record_rate_limit(client_id, "allowed")
            return RateLimitResult(
                allowed=True, remaining_tokens=int(tokens), retry_after_seconds=0
            )

        # Throttled: compute required wait time
        needed = tokens_needed - tokens
        retry_after = max(1, int(needed / self.refill_rate))
        self._memory_buckets[client_id] = (tokens, now)
        record_rate_limit(client_id, "throttled")
        return RateLimitResult(allowed=False, remaining_tokens=0, retry_after_seconds=retry_after)

    def _check_redis(self, client_id: str, tokens_needed: int, now: float) -> RateLimitResult:
        key = f"rate_limit:{client_id}"
        pipe = self.redis_client.pipeline()
        pipe.hget(key, "tokens")
        pipe.hget(key, "last_time")
        raw_tokens, raw_last_time = pipe.execute()

        tokens = float(raw_tokens) if raw_tokens else float(self.capacity)
        last_time = float(raw_last_time) if raw_last_time else now

        elapsed = now - last_time
        tokens = min(self.capacity, tokens + elapsed * self.refill_rate)

        if tokens >= tokens_needed:
            tokens -= tokens_needed
            pipe = self.redis_client.pipeline()
            pipe.hset(key, mapping={"tokens": str(tokens), "last_time": str(now)})
            pipe.expire(key, 300)
            pipe.execute()
            record_rate_limit(client_id, "allowed")
            return RateLimitResult(
                allowed=True, remaining_tokens=int(tokens), retry_after_seconds=0
            )

        needed = tokens_needed - tokens
        retry_after = max(1, int(needed / self.refill_rate))
        record_rate_limit(client_id, "throttled")
        return RateLimitResult(allowed=False, remaining_tokens=0, retry_after_seconds=retry_after)


# Global default rate limiter instance
rate_limiter = TokenBucketRateLimiter(capacity=20, refill_rate=2.0)
