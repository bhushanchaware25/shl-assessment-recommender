"""
retriever.py — Hybrid BM25 + semantic search for the SHL catalog.

Loads directly from shl_product_catalog.json (the official source file),
normalizing fields inline — no intermediate catalog.json needed.
"""

import json
import logging
import os
import re
from typing import List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CATALOG_PATH = os.environ.get("CATALOG_PATH", "shl_product_catalog.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BM25_WEIGHT = 0.35
SEMANTIC_WEIGHT = 0.65

# Key name → letter code mapping
_KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Biodata & Situational Judgement": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Personality & Behaviour": "P",
    "Simulations": "S",
}

# ── Module-level state ────────────────────────────────────────────────────────
_catalog: List[dict] = []
_bm25: Optional[BM25Okapi] = None
_embed_model: Optional[SentenceTransformer] = None
_catalog_embeddings: Optional[np.ndarray] = None
_url_set: set = set()


def _normalize(raw: dict) -> dict:
    """Convert one shl_product_catalog.json entry to the normalized schema."""
    keys_full = raw.get("keys", [])
    test_types = []
    for k in keys_full:
        code = _KEY_TO_CODE.get(k)
        if code and code not in test_types:
            test_types.append(code)

    duration_str = raw.get("duration", "") or ""
    m = re.search(r"(\d+)", duration_str)
    duration_int = int(m.group(1)) if m else None

    return {
        "name": raw["name"],
        "url": raw["link"],
        "description": raw.get("description", ""),
        "test_types": test_types,
        "test_types_full": keys_full,
        "duration": duration_int,
        "duration_str": duration_str,
        "remote_testing": raw.get("remote", "no").lower() == "yes",
        "adaptive_irt": raw.get("adaptive", "no").lower() == "yes",
        "job_levels": raw.get("job_levels", []),
        "languages": raw.get("languages", []),
    }


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer."""
    return text.lower().split()


def _build_corpus_text(entry: dict) -> str:
    """Build the searchable text incorporating all enriched fields."""
    parts = [entry.get("name", ""), entry.get("description", "")]
    parts.extend(entry.get("test_types_full", []))
    parts.extend(entry.get("job_levels", []))
    parts.extend(entry.get("languages", [])[:5])
    if entry.get("duration_str"):
        parts.append(entry["duration_str"])
    return " ".join(p for p in parts if p)


def initialize(catalog_path: str = CATALOG_PATH) -> None:
    """Load & normalize catalog, then build retrieval indices. Called once at startup."""
    global _catalog, _bm25, _embed_model, _catalog_embeddings, _url_set

    logger.info("Initializing retriever...")

    with open(catalog_path, encoding="utf-8", errors="replace") as f:
        raw_data = json.loads(f.read(), strict=False)

    _catalog = [_normalize(item) for item in raw_data]
    logger.info(f"Loaded and normalized {len(_catalog)} assessments from {catalog_path}")

    # Build URL set for anti-hallucination validation
    _url_set = {entry["url"].rstrip("/") for entry in _catalog}

    # Build BM25 index
    logger.info("Building BM25 index...")
    corpus = [_tokenize(_build_corpus_text(e)) for e in _catalog]
    _bm25 = BM25Okapi(corpus)

    # Load sentence transformer
    logger.info(f"Loading sentence-transformer: {EMBEDDING_MODEL}")
    _embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Compute catalog embeddings
    logger.info("Computing catalog embeddings...")
    texts = [_build_corpus_text(e) for e in _catalog]
    _catalog_embeddings = _embed_model.encode(texts, convert_to_numpy=True)

    logger.info(f"Retriever initialized. Catalog size: {len(_catalog)}")



def hybrid_search(query: str, top_k: int = 10) -> List[Tuple[dict, float]]:
    """
    Search the catalog using a hybrid of BM25 and semantic similarity.
    Returns a list of (entry, score) tuples sorted by score descending.
    """
    if not _catalog:
        raise RuntimeError("Retriever not initialized. Call initialize() first.")

    n = len(_catalog)

    # ── BM25 scores ──────────────────────────────────────────────────────────
    bm25_scores = np.array(_bm25.get_scores(_tokenize(query)))
    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_norm = bm25_scores / bm25_max
    else:
        bm25_norm = bm25_scores

    # ── Semantic scores ──────────────────────────────────────────────────────
    query_vec = _embed_model.encode([query], convert_to_numpy=True)[0]
    # Cosine similarity
    norms = np.linalg.norm(_catalog_embeddings, axis=1) * np.linalg.norm(query_vec)
    norms = np.where(norms == 0, 1e-10, norms)
    sem_scores = (_catalog_embeddings @ query_vec) / norms
    sem_norm = (sem_scores + 1) / 2  # shift from [-1,1] to [0,1]

    # ── Combined score ────────────────────────────────────────────────────────
    combined = BM25_WEIGHT * bm25_norm + SEMANTIC_WEIGHT * sem_norm

    # Top-k indices
    top_indices = np.argsort(combined)[::-1][:top_k]

    return [(_catalog[i], float(combined[i])) for i in top_indices]


def is_valid_catalog_url(url: str) -> bool:
    """Return True if the URL exists in the catalog (anti-hallucination gate)."""
    return url.rstrip("/") in _url_set


def get_all_assessments() -> List[dict]:
    """Return the full catalog."""
    return _catalog
    