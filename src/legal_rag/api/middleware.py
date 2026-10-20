"""Correlation ID and request logging middleware."""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from legal_rag.config import settings
from legal_rag.logging_conf import set_correlation_id

logger = logging.getLogger("legal_rag.api.middleware")


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Ensures every HTTP request has a traceable correlation ID attached to context and headers."""

    async def dispatch(self, request: Request, call_next) -> Response:
        corr_id = request.headers.get(settings.CORRELATION_HEADER) or str(uuid.uuid4())
        set_correlation_id(corr_id)

        start_time = time.time()
        logger.info(
            f"Incoming request: {request.method} {request.url.path}",
            extra={"path": request.url.path, "method": request.method},
        )

        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000
            response.headers[settings.CORRELATION_HEADER] = corr_id
            response.headers["X-Model-Version"] = settings.API_VERSION
            logger.info(
                f"Completed request: {request.method} {request.url.path} - status={response.status_code} in {duration_ms:.2f}ms",
                extra={"status_code": response.status_code, "duration_ms": duration_ms},
            )
            return response
        except Exception:
            duration_ms = (time.time() - start_time) * 1000
            logger.exception(
                f"Unhandled error processing {request.method} {request.url.path}",
                extra={"duration_ms": duration_ms},
            )
            raise
