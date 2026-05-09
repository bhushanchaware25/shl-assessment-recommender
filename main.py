"""
main.py — FastAPI application for the SHL Assessment Recommender.

Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → ChatResponse

Startup: loads shl_product_catalog.json and embedding model into memory.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import retriever
import agent as agent_module
from models import ChatRequest, ChatResponse, HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize retriever (catalog + embedding model) at startup."""
    logger.info("Starting up SHL Assessment Recommender...")
    try:
        retriever.initialize()
        logger.info("Startup complete — ready to serve requests.")
    except RuntimeError as e:
        # Fail loudly at startup, not at request time
        logger.critical(f"Startup failed: {e}")
        raise

    yield  # app is running

    logger.info("Shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "A conversational agent that helps hiring managers find the right "
        "SHL assessments for their roles. Powered by Groq (llama-3.3-70b-versatile)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for the evaluator
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "reply": (
                "An unexpected error occurred. Please try again or "
                "contact support if the issue persists."
            ),
            "recommendations": [],
            "end_of_conversation": False,
        },
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    """Service info and available endpoints."""
    return {
        "service": "SHL Assessment Recommender",
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "chat": "POST /chat",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns {"status": "ok"} when the service is ready.
    """
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Conversational endpoint for SHL assessment recommendations.

    Send a conversation history (with the latest user message last).
    Returns a reply, optional recommendations, and end_of_conversation flag.
    """
    messages = request.messages

    # Validate: last message must be from user
    if not messages or messages[-1].role != "user":
        raise HTTPException(
            status_code=422,
            detail="The last message in the conversation must have role='user'.",
        )

    # Convert to plain dicts for agent
    messages_dicts = [{"role": m.role, "content": m.content} for m in messages]

    try:
        ag = agent_module.get_agent()
        response = ag.process_turn(messages_dicts)
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        return ChatResponse(
            reply=(
                "I encountered an error while processing your request. "
                "Please try again."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    return response


# ── Dev server entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        reload=False,
        log_level="info",
    )
