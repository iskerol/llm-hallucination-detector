import time
import json
import logging
from collections import deque
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("api.middleware")
logger.setLevel(logging.INFO)

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Logs structured JSON HTTP analytics."""
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise e
        finally:
            process_time = time.time() - start_time
            log_data = {
                "method": request.method,
                "url": str(request.url),
                "status_code": status_code,
                "process_time_ms": round(process_time * 1000, 2),
                "client": request.client.host if request.client else "unknown"
            }
            logger.info(json.dumps(log_data))
        return response

class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Enriches output header trace tracking internal pipeline delays."""
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Lightweight in-memory sliding window bounding IP flood thresholds."""
    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.ip_window = {}
        
    async def dispatch(self, request: Request, call_next):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        if ip not in self.ip_window:
            self.ip_window[ip] = deque()
            
        # Drain expired window logs
        while self.ip_window[ip] and self.ip_window[ip][0] < now - self.window_seconds:
            self.ip_window[ip].popleft()
            
        if len(self.ip_window[ip]) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."}
            )
            
        self.ip_window[ip].append(now)
        return await call_next(request)

def setup_middlewares(app):
    """Initialize completely stacked middleware routes globally."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(StructuredLoggingMiddleware)
    app.add_middleware(RequestTimingMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)
