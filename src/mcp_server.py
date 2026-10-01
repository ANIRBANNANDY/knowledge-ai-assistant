"""
Model Context Protocol (MCP) Server for Knowledge AI Chatbot.
Exposes standard MCP tools over stdio for Antigravity (Gemini 3.7 Flash) to query
and manage the local documentation knowledge base without requiring external API keys.
Includes full JSON-RPC logging of all Gemini <-> MCP interactions.
"""

from __future__ import annotations
import io
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent import KnowledgeQueryAgent
from src.chunker import DocumentChunker
from src.llm import LocalLLMClient
from src.logger import log_event
from src.retriever import KnowledgeRetriever
from src.scraper import WebScraper
from src.storage import KnowledgeStorage

# Set UTF-8 encoding for stdio
if sys.platform == "win32":
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


class KnowledgeMCPServer:
    def __init__(self, db_path: str = "data/knowledge_base.db"):
        abs_db = os.path.join(PROJECT_ROOT, db_path) if not os.path.isabs(db_path) else db_path
        self.storage = KnowledgeStorage(abs_db)
        self.scraper = WebScraper()
        self.chunker = DocumentChunker()
        self.retriever = KnowledgeRetriever(self.storage)
        self.llm_client = LocalLLMClient()
        self.agent = KnowledgeQueryAgent(
            retriever=self.retriever, storage=self.storage, llm_client=self.llm_client
        )
        log_event(
            category="MCP",
            event_type="SERVER_START",
            action="MCP Server Initialized with Local LLM",
            details={"db_path": abs_db, "llm_model": self.llm_client.default_model},
        )


    def get_tools_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "query_knowledge_agent",
                "description": (
                    "[PRIMARY TOOL] Query the dedicated Multi-Stage Knowledge Agent for comprehensive, verified answers. "
                    "ALWAYS use this tool first for ANY documentation question. "
                    "Performs intent classification (Definition/Workflow/Contextual/Comparison/Troubleshooting/Benefits), "
                    "query decomposition, anti-hallucination verification, "
                    "and returns a fully structured, cited markdown answer. "
                    "Use this for: 'What is X?', 'How to configure Y?', 'What happens if I change Z?', "
                    "'What are the benefits of X?', 'Difference between X and Y?'"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The technical documentation question or topic.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of evidence chunks to analyze (default 6).",
                            "default": 6,
                        },
                        "use_llm": {
                            "type": "boolean",
                            "description": "Whether to use local Ollama LLM for grounded answer synthesis (default true).",
                            "default": True,
                        },
                        "model": {
                            "type": "string",
                            "description": "Specific Ollama model to use (default: gemma4:12b).",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_llm_status",
                "description": "Inspect local Ollama LLM server health, connection status, active model, and available models.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "search_knowledge_base",
                "description": (
                    "[SECONDARY TOOL] Raw hybrid search of the local documentation knowledge base. "
                    "Returns ranked text chunks with section breadcrumbs, source URLs, and scores. "
                    "Use this ONLY when you need raw evidence chunks for manual inspection, or when "
                    "query_knowledge_agent did not return a satisfactory answer and you want to see raw hits. "
                    "For all standard documentation questions, use query_knowledge_agent instead."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The natural language question or keyword query.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of ranked chunks to retrieve (default 5).",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_document",
                "description": "Retrieve the complete scraped Markdown content and metadata for a specific document URL.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the document to inspect.",
                        }
                    },
                    "required": ["url"],
                },
            },
            {
                "name": "list_knowledge_sources",
                "description": "List all indexed documentation URLs, titles, chunk counts, and word counts in the knowledge base.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "ingest_url",
                "description": "Scrape, chunk, and index a new documentation URL into the local knowledge base dynamically.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The web URL to scrape and index.",
                        }
                    },
                    "required": ["url"],
                },
            },
        ]

    def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        try:
            if name == "get_llm_status":
                health = self.llm_client.check_health()
                elapsed_ms = (time.time() - t0) * 1000
                log_event(
                    category="MCP",
                    event_type="TOOL_CALL_EXECUTED",
                    action="Agent checked local LLM status",
                    details=health,
                    latency_ms=elapsed_ms,
                )
                return {
                    "content": [{"type": "text", "text": json.dumps(health, indent=2)}],
                    "isError": False,
                }

            elif name == "query_knowledge_agent":
                query = arguments.get("query", "")
                top_k = int(arguments.get("top_k", 6))
                use_llm = arguments.get("use_llm", True)
                model = arguments.get("model", None)
                agent_resp = self.agent.answer_query(query, top_k=top_k, use_llm=use_llm, model=model)
                elapsed_ms = (time.time() - t0) * 1000

                log_event(
                    category="MCP",
                    event_type="TOOL_CALL_EXECUTED",
                    action=f"Agent executed 'query_knowledge_agent' ({agent_resp.engine}) for '{query[:50]}'",
                    details={
                        "tool": name,
                        "query": query,
                        "intent": agent_resp.intent,
                        "engine": agent_resp.engine,
                        "model": agent_resp.model,
                        "evidence_count": agent_resp.evidence_count,
                        "unindexed": agent_resp.unindexed,
                    },
                    latency_ms=elapsed_ms,
                )

                return {
                    "content": [{"type": "text", "text": agent_resp.reply}],
                    "isError": False,
                }


            elif name == "search_knowledge_base":
                query = arguments.get("query", "")
                top_k = int(arguments.get("top_k", 5))
                hits = self.retriever.hybrid_search(query, top_k=top_k)

                output_parts = [
                    f"# Search Results for: '{query}'\n",
                    f"**Total Hits Found:** {len(hits)}\n",
                    "> **Strict Grounding Rule:** Base your answer ONLY on these returned sections. Always cite URL links.\n",
                ]

                for i, hit in enumerate(hits):
                    output_parts.append(
                        f"### [{i+1}] {hit.doc_title}\n"
                        f"- **Section:** {hit.section_path}\n"
                        f"- **URL:** {hit.doc_url}\n"
                        f"- **Score:** {hit.score:.5f}\n\n"
                        f"{hit.content}\n"
                        f"---\n"
                    )

                result_text = "\n".join(output_parts)
                elapsed_ms = (time.time() - t0) * 1000

                log_event(
                    category="MCP",
                    event_type="TOOL_CALL_EXECUTED",
                    action=f"Gemini executed 'search_knowledge_base' for '{query[:50]}'",
                    details={
                        "tool": name,
                        "query": query,
                        "top_k": top_k,
                        "hits_count": len(hits),
                    },
                    payload={"hits": [h.model_dump() for h in hits[:2]]},
                    latency_ms=elapsed_ms,
                )

                return {
                    "content": [{"type": "text", "text": result_text}],
                    "isError": False,
                }

            elif name == "get_document":
                url = arguments.get("url", "")
                doc = self.storage.get_document(url)
                elapsed_ms = (time.time() - t0) * 1000

                if not doc:
                    log_event(
                        category="MCP",
                        event_type="TOOL_CALL_ERROR",
                        action=f"Gemini requested non-existent doc '{url}'",
                        details={"tool": name, "url": url},
                        status="error",
                        latency_ms=elapsed_ms,
                    )
                    return {
                        "content": [
                            {"type": "text", "text": f"Document not found for URL: {url}"}
                        ],
                        "isError": True,
                    }

                text = (
                    f"# {doc['title']}\n"
                    f"- **URL:** {doc['url']}\n"
                    f"- **Word Count:** {doc['word_count']}\n"
                    f"- **Chunks:** {doc['chunk_count']}\n"
                    f"- **Scraped At:** {doc['scraped_at']}\n\n"
                    f"## Content:\n\n{doc['markdown_content']}"
                )

                log_event(
                    category="MCP",
                    event_type="TOOL_CALL_EXECUTED",
                    action=f"Gemini retrieved full document '{doc['title'][:40]}'",
                    details={"tool": name, "url": url, "word_count": doc["word_count"]},
                    latency_ms=elapsed_ms,
                )

                return {
                    "content": [{"type": "text", "text": text}],
                    "isError": False,
                }

            elif name == "list_knowledge_sources":
                docs = self.storage.list_documents()
                elapsed_ms = (time.time() - t0) * 1000

                if not docs:
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": "The knowledge base is currently empty. Use `ingest_url` to add documentation.",
                            }
                        ],
                        "isError": False,
                    }

                lines = ["# Indexed Knowledge Sources\n"]
                lines.append(f"Total Documents: {len(docs)}\n")
                for i, d in enumerate(docs):
                    lines.append(
                        f"{i+1}. **{d['headline']}**\n"
                        f"   - URL: {d['url']}\n"
                        f"   - Chunks: {d['chunk_count']} | Words: {d['word_count']}\n"
                        f"   - Last Updated: {d['updated_at']}\n"
                    )

                log_event(
                    category="MCP",
                    event_type="TOOL_CALL_EXECUTED",
                    action=f"Gemini listed knowledge sources ({len(docs)} docs)",
                    details={"tool": name, "doc_count": len(docs)},
                    latency_ms=elapsed_ms,
                )

                return {
                    "content": [{"type": "text", "text": "\n".join(lines)}],
                    "isError": False,
                }

            elif name == "ingest_url":
                url = arguments.get("url", "")
                doc = self.scraper.scrape(url)
                chunks = self.chunker.chunk_document(doc)
                self.storage.save_document_and_chunks(doc, chunks)
                elapsed_ms = (time.time() - t0) * 1000

                msg = (
                    f"Successfully scraped and indexed URL: {url}\n"
                    f"- Title: {doc.title}\n"
                    f"- Headline: {doc.headline}\n"
                    f"- Word Count: {doc.word_count}\n"
                    f"- Chunks Created: {len(chunks)}"
                )

                # Invalidate vocabulary cache so new doc is immediately searchable
                self.retriever.invalidate_vocab_cache()

                log_event(
                    category="MCP",
                    event_type="TOOL_CALL_EXECUTED",
                    action=f"Gemini ingested URL '{url}'",
                    details={"tool": name, "url": url, "chunks": len(chunks)},
                    latency_ms=elapsed_ms,
                )

                return {
                    "content": [{"type": "text", "text": msg}],
                    "isError": False,
                }

            else:
                elapsed_ms = (time.time() - t0) * 1000
                log_event(
                    category="MCP",
                    event_type="TOOL_CALL_ERROR",
                    action=f"Unknown tool call '{name}'",
                    details={"tool": name},
                    status="error",
                    latency_ms=elapsed_ms,
                )
                return {
                    "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                    "isError": True,
                }

        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000
            log_event(
                category="MCP",
                event_type="TOOL_CALL_EXCEPTION",
                action=f"Exception in tool '{name}': {str(e)}",
                details={"tool": name, "error": str(e)},
                status="error",
                latency_ms=elapsed_ms,
            )
            return {
                "content": [{"type": "text", "text": f"Error executing tool '{name}': {str(e)}"}],
                "isError": True,
            }

    def run(self):
        """Run JSON-RPC 2.0 stdio server loop with event logging."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()
                continue

            req_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})

            # Handle JSON-RPC methods
            if method == "initialize":
                log_event(
                    category="MCP",
                    event_type="HANDSHAKE",
                    action="Gemini Language Server Connected (initialize)",
                    details={"client_params": params},
                )
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": "knowledge-chatbot-mcp",
                            "version": "1.1.0",
                        },
                    },
                }
            elif method == "notifications/initialized" or method == "initialized":
                log_event(
                    category="MCP",
                    event_type="NOTIFICATION",
                    action="Gemini initialized notification received",
                    details={},
                )
                continue
            elif method == "ping":
                response = {"jsonrpc": "2.0", "id": req_id, "result": {}}
            elif method == "tools/list":
                log_event(
                    category="MCP",
                    event_type="DISCOVERY",
                    action="Gemini requested tools list",
                    details={},
                )
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": self.get_tools_list()},
                }
            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {})
                res = self.handle_tool_call(tool_name, tool_args)
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": res,
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method '{method}' not found",
                    },
                }

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    server = KnowledgeMCPServer()
    server.run()
