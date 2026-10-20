from time import perf_counter

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send


_HTTP_METHODS = frozenset(
    {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT"}
)


class HTTPMonitoring:
    def __init__(self):
        self.registry = CollectorRegistry()

        self.requests = Counter(
            "company_sentiment_http_requests_total",
            "Total HTTP requests by endpoint, method and status code.",
            labelnames=("endpoint", "method", "status_code"),
            registry=self.registry,
        )

        self.duration = Histogram(
            "company_sentiment_http_request_duration_seconds",
            "HTTP response duration in seconds.",
            labelnames=("endpoint", "method"),
            buckets=(
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1,
                2.5,
                5,
                10,
                30,
                60,
                120,
            ),
            registry=self.registry,
        )

    def observe_request(self, scope: Scope, status_code: int, duration: float) -> None:
        route = scope.get("route")
        endpoint = getattr(route, "path", None) or "unmatched"

        method = scope.get("method", "OTHER").upper()
        if method not in _HTTP_METHODS:
            method = "OTHER"

        self.requests.labels(
            endpoint=endpoint,
            method=method,
            status_code=str(status_code),
        ).inc()

        self.duration.labels(
            endpoint=endpoint,
            method=method,
        ).observe(duration)

    def metrics_response(self) -> Response:
        return Response(
            content=generate_latest(self.registry),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )


class HTTPMetricsMiddleware:
    def __init__(self, app: ASGIApp, monitoring: HTTPMonitoring):
        self.app = app
        self.monitoring = monitoring

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope.get("path", "").rstrip("/") == "/metrics":
            await self.app(scope, receive, send)
            return

        started = perf_counter()
        status_code = 500
        recorded = False

        def record_request() -> None:
            nonlocal recorded

            if recorded:
                return

            recorded = True
            self.monitoring.observe_request(
                scope,
                status_code,
                perf_counter() - started,
            )

        async def monitored_send(message: Message) -> None:
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]

            await send(message)

            if message["type"] == "http.response.body":
                if not message.get("more_body", False):
                    record_request()

        try:
            await self.app(scope, receive, monitored_send)
        finally:
            record_request()
