"""
One-click launcher for the Knowledge AI Chatbot server & web dashboard.
Auto-seeds initial Bitbucket documentation if the knowledge base is empty.
"""

from __future__ import annotations
import os
import sys
import uvicorn

# Configure UTF-8 encoding for Windows terminals
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from src.chunker import DocumentChunker
from src.scraper import WebScraper
from src.storage import KnowledgeStorage

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URLS = [
    "https://support.atlassian.com/bitbucket-cloud/docs/what-is-a-workspace/",
    "https://support.atlassian.com/bitbucket-cloud/docs/create-your-workspace/",
    "https://support.atlassian.com/bitbucket-cloud/docs/grant-access-to-a-workspace/",
]


def ensure_seeded():
    db_path = os.path.join(PROJECT_ROOT, "data", "knowledge_base.db")
    storage = KnowledgeStorage(db_path)
    docs = storage.list_documents()

    if not docs:
        print("[*] Knowledge base is empty. Seeding default Bitbucket Cloud documentation...")
        scraper = WebScraper()
        chunker = DocumentChunker()
        for url in DEFAULT_URLS:
            try:
                print(f"    --> Scraping: {url}")
                doc = scraper.scrape(url)
                chunks = chunker.chunk_document(doc)
                storage.save_document_and_chunks(doc, chunks)
                print(f"        [OK] Ingested '{doc.headline}' ({len(chunks)} chunks)")
            except Exception as e:
                print(f"        [!] Error scraping {url}: {e}")
        print("[*] Initial knowledge seeding complete!\n")
    else:
        print(f"[*] Knowledge base ready: {len(docs)} documents loaded.\n")


def main():
    ensure_seeded()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))

    print("=" * 70)
    print("  [>] KNOWLEDGE AI CHATBOT SERVER STARTING")
    print(f"  [>] Web Dashboard: http://{host}:{port}")
    print(f"  [>] Swagger API Docs: http://{host}:{port}/docs")
    print("  [>] Knowledge Agent: Dedicated Multi-Stage Agent Active")
    print("=" * 70 + "\n")

    uvicorn.run("src.api:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
