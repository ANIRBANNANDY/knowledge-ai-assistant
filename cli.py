"""
Command-Line Interface (CLI) for Knowledge AI Chatbot.
Manage URLs, inspect chunks, test hybrid search, and interactive terminal chat.
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import List

from src.chunker import DocumentChunker
from src.retriever import KnowledgeRetriever
from src.scraper import WebScraper
from src.storage import KnowledgeStorage

DEFAULT_URLS = [
    "https://support.atlassian.com/bitbucket-cloud/docs/what-is-a-workspace/",
    "https://support.atlassian.com/bitbucket-cloud/docs/create-your-workspace/",
    "https://support.atlassian.com/bitbucket-cloud/docs/grant-access-to-a-workspace/",
]


def get_components(db_path: str = "data/knowledge_base.db"):
    from src.agent import KnowledgeQueryAgent
    from src.llm import LocalLLMClient
    storage = KnowledgeStorage(db_path)
    scraper = WebScraper()
    chunker = DocumentChunker()
    retriever = KnowledgeRetriever(storage)
    llm_client = LocalLLMClient()
    agent = KnowledgeQueryAgent(retriever=retriever, storage=storage, llm_client=llm_client)
    return storage, scraper, chunker, retriever, agent, llm_client


def cmd_list(args):
    storage, _, _, _, _, _ = get_components(args.db)
    docs = storage.list_documents()
    print("\n" + "=" * 70)
    print(f"  INDEXED KNOWLEDGE SOURCES ({len(docs)} Documents)")
    print("=" * 70)
    if not docs:
        print("  (Knowledge base is empty. Run: python cli.py seed)")
    for i, d in enumerate(docs, 1):
        print(f"\n[{i}] {d['headline']}")
        print(f"    URL:      {d['url']}")
        print(f"    Chunks:   {d['chunk_count']} | Words: {d['word_count']}")
        print(f"    Updated:  {d['updated_at']}")
    print("\n" + "=" * 70 + "\n")


def cmd_ingest(args):
    storage, scraper, chunker, _, _, _ = get_components(args.db)
    urls = args.urls if args.urls else DEFAULT_URLS
    print(f"\nIngesting {len(urls)} URL(s)...")
    for url in urls:
        try:
            print(f"\n--> Fetching: {url}")
            doc = scraper.scrape(url)
            chunks = chunker.chunk_document(doc)
            storage.save_document_and_chunks(doc, chunks)
            print(f"    [OK] Ingested '{doc.headline}'")
            print(f"         {len(chunks)} chunks created ({doc.word_count} words)")
        except Exception as e:
            print(f"    [ERROR] Failed to ingest {url}: {e}")
    print("\nDone!\n")


def cmd_delete(args):
    storage, _, _, _, _, _ = get_components(args.db)
    success = storage.delete_document(args.url)
    if success:
        print(f"\n[OK] Deleted document and its chunks: {args.url}\n")
    else:
        print(f"\n[WARN] Document not found: {args.url}\n")


def cmd_llm(args):
    from src.llm import LocalLLMClient
    client = LocalLLMClient()
    health = client.check_health()
    print("\n" + "=" * 70)
    print("  LOCAL LLM STATUS (OLLAMA)")
    print("=" * 70)
    print(f"  Base URL:        {health.get('base_url', 'http://localhost:11434')}")
    print(f"  Available:       {'ONLINE (Connected)' if health.get('available') else 'OFFLINE'}")
    print(f"  Active Model:    {health.get('active_model')}")
    print(f"  Installed Models:")
    models = health.get("models", [])
    if models:
        for m in models:
            active_marker = " <-- ACTIVE" if m == health.get("active_model") else ""
            print(f"    - {m}{active_marker}")
    else:
        print("    (No models found in Ollama)")
    if health.get("error"):
        print(f"\n  Notice: {health['error']}")
    print("=" * 70 + "\n")


def cmd_query(args):
    from src.retriever import SearchHit
    _, _, _, retriever, agent, _ = get_components(args.db)
    use_llm = not getattr(args, "no_llm", False)
    model = getattr(args, "model", None)
    think = getattr(args, "think", False)

    print(f"\nSearching & synthesizing answer (Engine: {'Ollama Local LLM' if use_llm else 'Rule-based'}, Mode: {'Deep Reasoning' if think else 'Turbo Direct'})...")
    agent_resp = agent.answer_query(args.question, top_k=args.top_k, use_llm=use_llm, model=model, think=think)

    print("\n" + "=" * 75)
    print(f"  KNOWLEDGE QUERY AGENT ANSWER")
    print(f"  Query:  '{args.question}' | Intent: {agent_resp.intent}")
    print(f"  Engine: {agent_resp.engine} | Evidence Chunks: {agent_resp.evidence_count}")
    if agent_resp.eval_count:
        print(f"  Tokens: {agent_resp.eval_count} generated (prompt: {agent_resp.prompt_eval_count})")
    print("=" * 75)
    print("\n" + agent_resp.reply + "\n")
    print("=" * 75)

    if args.show_prompt:
        formatted = retriever.format_context_for_llm(args.question, [SearchHit(**h) for h in agent_resp.hits])
        print("\n" + "=" * 75)
        print("  GROUNDED CONTEXT PROMPT:")
        print("=" * 75)
        print(formatted["prompt"])
        print("=" * 75 + "\n")


def cmd_interactive_chat(args):
    storage, _, _, _, agent, llm_client = get_components(args.db)
    docs = storage.list_documents()
    health = llm_client.check_health()
    use_llm = not getattr(args, "no_llm", False)
    current_model = getattr(args, "model", None) or llm_client.default_model

    print("\n" + "=" * 70)
    print("  KNOWLEDGE AI ASSISTANT - TERMINAL CHAT")
    print(f"  Active Knowledge Base: {len(docs)} documents loaded")
    if health.get("available") and use_llm:
        print(f"  Local LLM: Active ({current_model}) via Ollama")
    else:
        print("  Synthesis Engine: Rule-Based Deterministic")
    print("  Commands: /llm on|off, /model <name>, exit / quit")
    print("=" * 70 + "\n")

    while True:
        try:
            query = input(f"Ask [{current_model if use_llm else 'RuleBased'}] > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            if query.lower() == "/llm on":
                use_llm = True
                print(f"[OK] Local LLM enabled ({current_model})")
                continue
            if query.lower() == "/llm off":
                use_llm = False
                print("[OK] Local LLM disabled (using Rule-Based mode)")
                continue
            if query.lower().startswith("/model "):
                current_model = query.split(maxsplit=1)[1].strip()
                print(f"[OK] Model switched to: {current_model}")
                continue

            agent_resp = agent.answer_query(query, top_k=args.top_k, use_llm=use_llm, model=current_model)
            print("\n" + "-" * 70)
            print(f"[{agent_resp.engine} | {agent_resp.intent}] ({agent_resp.evidence_count} evidence chunks)\n")
            print(agent_resp.reply)
            print("-" * 70 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Local Knowledge AI Chatbot - Bitbucket Cloud Docs"
    )
    parser.add_argument("--db", default="data/knowledge_base.db", help="Path to SQLite database")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list
    subparsers.add_parser("list", help="List all indexed sources")

    # ingest / seed
    p_ingest = subparsers.add_parser("ingest", help="Scrape and index URLs")
    p_ingest.add_argument("urls", nargs="*", help="Web URLs to scrape and index")

    subparsers.add_parser("seed", help="Ingest default 3 Bitbucket URLs")

    # delete
    p_delete = subparsers.add_parser("delete", help="Delete a document by URL")
    p_delete.add_argument("url", help="URL of document to delete")

    # llm status
    subparsers.add_parser("llm", help="Check local Ollama LLM status and models")

    # query
    p_query = subparsers.add_parser("query", help="Search the knowledge base")
    p_query.add_argument("question", help="Question or query string")
    p_query.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve (default 3)")
    p_query.add_argument("--model", "-m", help="Specific Ollama model to use (default: gemma4:12b)")
    p_query.add_argument("--think", action="store_true", help="Enable Deep Reasoning / Chain of Thought mode")
    p_query.add_argument("--no-llm", action="store_true", help="Bypass LLM and use deterministic rule-based synthesis")
    p_query.add_argument("--show-prompt", action="store_true", help="Print full formatted LLM prompt")

    # chat
    p_chat = subparsers.add_parser("chat", help="Start interactive terminal chat")
    p_chat.add_argument("--top-k", type=int, default=3, help="Number of chunks per query")
    p_chat.add_argument("--model", "-m", help="Specific Ollama model to use (default: gemma4:12b)")
    p_chat.add_argument("--think", action="store_true", help="Enable Deep Reasoning / Chain of Thought mode")
    p_chat.add_argument("--no-llm", action="store_true", help="Start in rule-based mode without LLM")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "seed":
        args.urls = DEFAULT_URLS
        cmd_ingest(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "llm":
        cmd_llm(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "chat":
        cmd_interactive_chat(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

