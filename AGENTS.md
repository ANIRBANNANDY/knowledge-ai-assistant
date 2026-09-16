# Knowledge AI Chatbot Workspace

This workspace hosts a local Python-based Documentation Knowledge AI system.

## Key Subsystems
- `src/scraper.py`: High-fidelity web scraper extracting clean Markdown, JSON-LD structured data, tables, and headings.
- `src/chunker.py`: Hierarchical markdown chunker preserving section breadcrumbs, overlap windows, and token estimates.
- `src/storage.py`: SQLite database (`data/knowledge_base.db`) with FTS5 Okapi BM25 full-text indexing.
- `src/retriever.py`: Local hybrid search combining BM25 and TF-IDF vector cosine similarity using Reciprocal Rank Fusion (RRF).
- `src/mcp_server.py`: Model Context Protocol (MCP) server providing stdio tools to Gemini 3.7 Flash.
- `src/api.py`: FastAPI server for the Web UI and REST API.
- `static/`: Interactive Web Dashboard with Dark Glassmorphism aesthetic.
- `cli.py`: Command-line tool for ingestion, listing, and queries.

## Assistant Persona
- Ground all responses on retrieved local documentation.
- Provide direct citations with exact URLs and section titles.
