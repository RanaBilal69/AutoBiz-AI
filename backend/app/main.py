import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.core.rate_limiter import limiter, rate_limit_exceeded_handler
from app.core.logger import logger, log_api_request

load_dotenv()

security = HTTPBearer()

app = FastAPI(
    title="AI-Powered Business Automation Platform",
    description="Multi-agent system with Manager, Sales, and Inventory agents",
    version="1.0.0",
    swagger_ui_parameters={"persistAuthorization": True},
)

# ── Rate limiter ──────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ── CORS ──────────────────────────────────────────────────────
# List every origin that needs access — Vercel preview URLs use wildcards
# but FastAPI needs explicit origins when allow_credentials=True.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "https://autobiz-ai-rho.vercel.app",
    # Add any Vercel preview URL you use, e.g.:
    # "https://autobiz-ai-git-main-yourname.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request logging ───────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    user = request.headers.get("X-User", "anonymous")
    log_api_request(
        method=request.method,
        path=str(request.url.path),
        user=user,
        status=response.status_code,
    )
    return response

# ── Routers ───────────────────────────────────────────────────
from app.routers import auth, inventory, sales, manager, purchase_orders, stream, db_setup
app.include_router(auth.router)
app.include_router(inventory.router)
app.include_router(sales.router)
app.include_router(manager.router)
app.include_router(purchase_orders.router)
app.include_router(stream.router)
app.include_router(db_setup.router)

# ── Startup: create tables safely — never crash if DB is down ─
@app.on_event("startup")
async def startup_event():
    try:
        from app.database import get_engine
        from app.models import Base
        eng = get_engine()
        Base.metadata.create_all(bind=eng)
        logger.info("Database tables verified/created at startup.")
    except Exception as e:
        # Log the error but DO NOT crash — app still serves health checks
        logger.error(
            f"Startup DB connection failed: {e}\n"
            "Server is running but database is unavailable. "
            "Check DATABASE_URL in environment variables."
        )

# ── Health endpoints ──────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "running", "message": "AI Business Automation Platform"}

@app.get("/health")
def health():
    """Always returns 200 — used by HuggingFace to keep the container alive."""
    return {"status": "ok", "version": "1.0.0"}

@app.get("/health/db")
def health_db():
    """Check live DB connectivity — separate from app health."""
    try:
        from app.database import get_engine
        from sqlalchemy import text
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "connected"}
    except Exception as e:
        return {"status": "disconnected", "error": str(e)}, 503