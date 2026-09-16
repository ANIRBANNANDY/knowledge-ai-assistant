"""
FastAPI application backend for Knowledge AI Chatbot.
Provides REST APIs for search, document management, URL ingestion, chat,
and real-time activity/MCP process logging.
"""

from __future__ import annotations
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

# Configure UTF-8 encoding for stdio on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent import KnowledgeQueryAgent
from src.chunker import DocumentChunker
from src.logger import clear_logs, get_recent_logs, log_event
from src.retriever import KnowledgeRetriever
from src.scraper import WebScraper
from src.storage import KnowledgeStorage

app = FastAPI(
    title="Knowledge AI Chatbot API",
    description="Local RAG & Knowledge Retrieval for Documentation with Process Logging",
    version="1.2.0",
)

# Global Exception Handler to ensure JSON responses on errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log_event(
        category="WEB",
        event_type="UNHANDLED_EXCEPTION",
        action=f"Error processing {request.method} {request.url.path}: {str(exc)}",
        details={"path": str(request.url.path), "error": str(exc)},
        status="error",
    )
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "error": "Internal Server Error", "message": str(exc)},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error": exc.detail},
    )

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons
storage = KnowledgeStorage(os.path.join(PROJECT_ROOT, "data/knowledge_base.db"))
scraper = WebScraper()
chunker = DocumentChunker()
retriever = KnowledgeRetriever(storage)
knowledge_agent = KnowledgeQueryAgent(retriever=retriever, storage=storage)


# --- Request/Response Models ---
class IngestRequest(BaseModel):
    urls: List[str]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 6


class ChatRequest(BaseModel):
    message: str
    top_k: int = 6


@app.get("/api/status")
def get_status():
    docs = storage.list_documents()
    total_chunks = sum(d.get("chunk_count", 0) for d in docs)
    total_words = sum(d.get("word_count", 0) for d in docs)
    return {
        "status": "online",
        "total_documents": len(docs),
        "total_chunks": total_chunks,
        "total_words": total_words,
        "model_integration": "Dedicated Multi-Stage Knowledge Query Agent",
    }


@app.get("/api/documents")
def list_documents():
    return storage.list_documents()


@app.get("/api/documents/detail")
def get_document_detail(url: str = Query(..., description="Document URL")):
    doc = storage.get_document(url)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.post("/api/ingest")
def ingest_urls(req: IngestRequest):
    t0 = time.time()
    results = []
    for url in req.urls:
        url = url.strip()
        if not url:
            continue
        try:
            doc = scraper.scrape(url)
            chunks = chunker.chunk_document(doc)
            storage.save_document_and_chunks(doc, chunks)
            retriever.invalidate_vocab_cache()  # Ensure new doc is immediately searchable
            results.append(
                {
                    "url": url,
                    "status": "success",
                    "title": doc.title,
                    "headline": doc.headline,
                    "chunks": len(chunks),
                    "word_count": doc.word_count,
                }
            )
        except Exception as e:
            results.append({"url": url, "status": "error", "error": str(e)})

    elapsed_ms = (time.time() - t0) * 1000
    log_event(
        category="WEB",
        event_type="WEB_INGEST",
        action=f"User ingested {len(req.urls)} URL(s)",
        details={"urls": req.urls, "results": results},
        latency_ms=elapsed_ms,
    )
    return {"results": results}


@app.delete("/api/documents")
def delete_document(url: str = Query(..., description="Document URL to delete")):
    deleted = storage.delete_document(url)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    log_event(
        category="WEB",
        event_type="WEB_DOC_DELETE",
        action=f"Deleted document: {url}",
        details={"url": url},
    )
    return {"status": "deleted", "url": url}


@app.post("/api/search")
def search_knowledge(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    t0 = time.time()
    hits = retriever.hybrid_search(req.query, top_k=req.top_k)
    formatted = retriever.format_context_for_llm(req.query, hits)
    elapsed_ms = (time.time() - t0) * 1000

    log_event(
        category="WEB",
        event_type="WEB_SEARCH",
        action=f"Search Query: '{req.query[:60]}'",
        details={"query": req.query, "top_k": req.top_k, "hits": len(hits)},
        latency_ms=elapsed_ms,
    )

    return {
        "query": req.query,
        "total_hits": len(hits),
        "hits": [hit.model_dump() for hit in hits],
        "citations": formatted["citations"],
        "prompt": formatted["prompt"],
        "context_markdown": formatted["context_markdown"],
    }


@app.post("/api/chat")
def chat_with_knowledge(req: ChatRequest):
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    t0 = time.time()
    try:
        agent_response = knowledge_agent.answer_query(query, top_k=req.top_k)
        elapsed_ms = (time.time() - t0) * 1000

        log_event(
            category="WEB",
            event_type="WEB_CHAT",
            action=f"Chat answered by Knowledge Agent: '{query[:50]}'",
            details={
                "query": query,
                "intent": agent_response.intent,
                "evidence_count": agent_response.evidence_count,
                "unindexed": agent_response.unindexed,
            },
            latency_ms=elapsed_ms,
        )

        return {
            "reply": agent_response.reply,
            "query": agent_response.query,
            "intent": agent_response.intent,
            "evidence_count": agent_response.evidence_count,
            "citations": agent_response.citations,
            "steps": [s.model_dump() for s in agent_response.steps],
            "hits": agent_response.hits,
            "unindexed": agent_response.unindexed,
        }
    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        log_event(
            category="WEB",
            event_type="WEB_CHAT_ERROR",
            action=f"Chat error for '{query[:40]}': {str(e)}",
            details={"query": query, "error": str(e)},
            status="error",
            latency_ms=elapsed_ms,
        )
        raise HTTPException(status_code=500, detail=f"Failed to process chat query: {str(e)}")


# --- Process & Activity Logs Endpoints ---
@app.get("/api/logs")
def get_logs(
    limit: int = Query(100, ge=1, le=500),
    category: Optional[str] = Query(None, description="Filter by category: MCP, WEB, RAG, SCRAPER"),
):
    return {"logs": get_recent_logs(limit=limit, category=category)}


@app.delete("/api/logs")
def delete_logs():
    success = clear_logs()
    return {"status": "cleared" if success else "error"}


# Serve static web frontend
static_dir = os.path.join(PROJECT_ROOT, "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
