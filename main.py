import os
import time
import uuid
import asyncio
from typing import Literal, Optional, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

# FastAPI and Security
from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# LLM and Resilience
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pybreaker

# Observability and Monitoring
import structlog
import prometheus_client as prom
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import httpx

# Configuration
from config import settings

# --- STRUCTURED LOGGING SETUP ---
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if settings.LOG_FORMAT == "json" else structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("SemanticRouter")

# --- METRICS SETUP ---
if settings.METRICS_ENABLED:
    REQUEST_COUNT = Counter('semantic_router_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
    REQUEST_DURATION = Histogram('semantic_router_request_duration_seconds', 'Request duration')
    ACTIVE_REQUESTS = Gauge('semantic_router_active_requests', 'Active requests')
    LLM_CALLS = Counter('semantic_router_llm_calls_total', 'LLM calls', ['model', 'status'])
    CIRCUIT_BREAKER_STATE = Gauge('semantic_router_circuit_breaker_state', 'Circuit breaker state')

# --- RATE LIMITING ---
limiter = Limiter(key_func=get_remote_address)

# --- CIRCUIT BREAKER ---
circuit_breaker = pybreaker.CircuitBreaker(
    fail_max=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    reset_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT
)

# --- EXCEPTION HANDLING ---
class SemanticRouterException(Exception):
    """Base exception for semantic router"""
    pass

class LLMProviderException(SemanticRouterException):
    """LLM provider specific exceptions"""
    pass

class ConfigurationException(SemanticRouterException):
    """Configuration related exceptions"""
    pass

# Initialize Async Client (Crucial for production performance)
try:
    if settings.LOCAL_LLM_URL:
        logger.info("MODE", mode="LOCAL_LLM", url=settings.LOCAL_LLM_URL)
        client = AsyncOpenAI(
            api_key="ollama", 
            base_url=settings.LOCAL_LLM_URL,
            timeout=settings.REQUEST_TIMEOUT
        )
    else:
        if not settings.OPENAI_API_KEY:
            raise ConfigurationException("OPENAI_API_KEY is missing for Cloud mode")
        logger.info("MODE", mode="CLOUD_OPENAI")
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=settings.REQUEST_TIMEOUT)
except Exception as e:
    logger.error("CLIENT_INITIALIZATION_FAILED", error=str(e))
    raise ConfigurationException(f"Failed to initialize LLM client: {e}")

# --- DATA MODELS ---
class RouteRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="User message to route")
    request_id: Optional[str] = None

class RouteResponse(BaseModel):
    route: Literal["simple_support", "complex_task", "error"]
    response: Optional[str]
    model_used: str
    processing_time_ms: float
    retry_count: int = 0
    request_id: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    mode: str
    circuit_breaker: str
    uptime_seconds: float
    version: str
    dependencies: Dict[str, str]

# --- RESISTANCE LAYER: ENHANCED RETRY LOGIC ---
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((LLMProviderException, Exception)),
    before_sleep=lambda retry_state: logger.warning(
        "RETRY_ATTEMPT",
        attempt=retry_state.attempt_number,
        max_attempts=3,
        model=retry_state.args[1] if retry_state.args else "unknown"
    )
)
async def resilient_llm_call(messages: list, model: str, request_id: str):
    """Enhanced LLM call with circuit breaker and detailed logging"""
    try:
        with circuit_breaker:
            logger.info(
                "LLM_CALL_START",
                model=model,
                message_count=len(messages),
                request_id=request_id
            )
            
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7 if model == settings.SMART_MODEL else 0.0,
                max_tokens=1024
            )
            
            result = response.choices[0].message.content
            
            logger.info(
                "LLM_CALL_SUCCESS",
                model=model,
                response_length=len(result) if result else 0,
                request_id=request_id
            )
            
            if settings.METRICS_ENABLED:
                LLM_CALLS.labels(model=model, status="success").inc()
            
            return result
            
    except Exception as e:
        logger.error(
            "LLM_CALL_FAILED",
            model=model,
            error=str(e),
            request_id=request_id
        )
        
        if settings.METRICS_ENABLED:
            LLM_CALLS.labels(model=model, status="error").inc()
        
        raise LLMProviderException(f"LLM call failed for model {model}: {e}")

# --- APP LIFECYCLE ---
start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("APPLICATION_STARTUP", version="2.0.0")
    yield
    logger.info("APPLICATION_SHUTDOWN")

app = FastAPI(
    title="Enterprise Semantic Router",
    description="Production-Grade Resilient, Async, Multi-Provider Intent Router",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.LOG_LEVEL == "DEBUG" else None
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Enterprise Semantic Router",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics", 
            "route": "/v1/route"
        },
        "docs": "/docs" if settings.LOG_LEVEL == "DEBUG" else "Disabled",
        "live_demo": "https://semantic-router-enterprise.onrender.com"
    }

# --- MIDDLEWARE SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure for production
)

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Add request context to logger
    logger = structlog.get_logger("SemanticRouter").bind(
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host
    )
    
    if settings.METRICS_ENABLED:
        ACTIVE_REQUESTS.inc()
    
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        
        logger.info(
            "REQUEST_COMPLETED",
            status_code=response.status_code,
            processing_time_ms=process_time
        )
        
        if settings.METRICS_ENABLED:
            REQUEST_COUNT.labels(
                method=request.method, 
                endpoint=request.url.path, 
                status=response.status_code
            ).inc()
            REQUEST_DURATION.observe(process_time / 1000)
        
        response.headers["X-Request-ID"] = request_id
        return response
        
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(
            "REQUEST_FAILED",
            error=str(e),
            processing_time_ms=process_time
        )
        
        if settings.METRICS_ENABLED:
            REQUEST_COUNT.labels(
                method=request.method, 
                endpoint=request.url.path, 
                status=500
            ).inc()
        
        raise
    finally:
        if settings.METRICS_ENABLED:
            ACTIVE_REQUESTS.dec()

# --- RATE LIMITING EXCEPTION HANDLER ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORE LOGIC ---

async def get_route(message: str, request_id: str) -> str:
    """Enhanced route classification with better error handling"""
    prompt = f"{settings.CLASSIFIER_PROMPT}\n\nInput: {message}\nLabel:"
    
    try:
        result = await resilient_llm_call(
            [{"role": "user", "content": prompt}], 
            settings.CLASSIFIER_MODEL,
            request_id
        )
        
        # Enhanced output cleaning
        result = result.lower().strip()
        if "simple" in result: 
            logger.info("ROUTE_CLASSIFIED", route="simple_support", request_id=request_id)
            return "simple_support"
        if "complex" in result: 
            logger.info("ROUTE_CLASSIFIED", route="complex_task", request_id=request_id)
            return "complex_task"
        
        logger.warning("ROUTE_FALLBACK", original_result=result, request_id=request_id)
        return "simple_support"
        
    except Exception as e:
        logger.error("ROUTE_CLASSIFICATION_FAILED", error=str(e), request_id=request_id)
        return "error"

@app.post("/v1/route", response_model=RouteResponse)
@limiter.limit(f"{settings.RATE_LIMIT_REQUESTS}/{settings.RATE_LIMIT_WINDOW}")
async def route_message(request: RouteRequest, http_request: Request):
    """Enhanced routing endpoint with comprehensive monitoring"""
    request_id = request.request_id or str(uuid.uuid4())
    start_time = time.time()
    
    logger = structlog.get_logger("SemanticRouter").bind(
        request_id=request_id,
        message_length=len(request.message)
    )
    
    try:
        # 1. Classify
        route = await get_route(request.message, request_id)
        
        if route == "error":
            raise LLMProviderException("Route classification failed")
        
        # 2. Execute
        model_to_use = settings.FAST_MODEL if route == "simple_support" else settings.SMART_MODEL
        system_prompt = (
            "You are a helpful assistant." 
            if route == "simple_support" 
            else "You are an expert technical assistant."
        )

        response_text = await resilient_llm_call(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ],
            model_to_use,
            request_id
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            "REQUEST_SUCCESS",
            route=route,
            model=model_to_use,
            processing_time_ms=processing_time
        )
        
        response = RouteResponse(
            route=route,
            response=response_text,
            model_used=model_to_use,
            processing_time_ms=processing_time,
            retry_count=0,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
        return JSONResponse(
            content=response.dict(),
            headers={"X-Router-Path": route}
        )
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(
            "REQUEST_FAILED",
            error=str(e),
            processing_time_ms=processing_time
        )
        
        return RouteResponse(
            route="error",
            response="I apologize, but I am experiencing technical difficulties connecting to the AI provider. Please try again.",
            model_used="None",
            processing_time_ms=processing_time,
            retry_count=3,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat()
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with detailed status"""
    uptime = time.time() - start_time
    circuit_state = "closed" if circuit_breaker.state == "closed" else "open"
    
    # Check external dependencies
    dependencies = {}
    try:
        if settings.LOCAL_LLM_URL:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{settings.LOCAL_LLM_URL}/tags")
                dependencies["ollama"] = "healthy" if response.status_code == 200 else "unhealthy"
        else:
            dependencies["openai"] = "configured"
    except Exception as e:
        dependencies["external"] = f"error: {str(e)}"
    
    if settings.METRICS_ENABLED:
        CIRCUIT_BREAKER_STATE.set(1 if circuit_breaker.state == "open" else 0)
    
    return HealthResponse(
        status="healthy",
        mode="local" if settings.LOCAL_LLM_URL else "cloud",
        circuit_breaker=circuit_state,
        uptime_seconds=uptime,
        version="2.0.0",
        dependencies=dependencies
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if not settings.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    
    return Response(generate_latest(), media_type="text/plain")

# --- EXCEPTION HANDLERS ---
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning("RATE_LIMIT_EXCEEDED", client_ip=request.client.host)
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )

@app.exception_handler(ConfigurationException)
async def config_handler(request: Request, exc: ConfigurationException):
    logger.error("CONFIGURATION_ERROR", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Server configuration error"}
    )

@app.exception_handler(LLMProviderException)
async def llm_handler(request: Request, exc: LLMProviderException):
    logger.error("LLM_PROVIDER_ERROR", error=str(exc))
    return JSONResponse(
        status_code=503,
        content={"detail": "AI provider temporarily unavailable"}
    )