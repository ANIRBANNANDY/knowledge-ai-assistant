"""
SQLite persistence layer with FTS5 (Full-Text Search) for documents and chunks.
"""

from __future__ import annotations
import json
import os
import re
import sqlite3
from typing import Dict, List, Optional
from src.chunker import DocumentChunk
from src.scraper import ScrapedDocument


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class KnowledgeStorage:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            resolved_db = os.path.join(PROJECT_ROOT, "data", "knowledge_base.db")
        elif not os.path.isabs(db_path):
            resolved_db = os.path.join(PROJECT_ROOT, db_path)
        else:
            resolved_db = db_path

        self.db_path = os.path.abspath(resolved_db)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            # 1. Documents table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    url TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    headline TEXT NOT NULL,
                    description TEXT,
                    markdown_content TEXT,
                    raw_text TEXT,
                    breadcrumbs TEXT,
                    word_count INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    scraped_at TEXT,
                    updated_at TEXT
                )
                """
            )

            # 2. Chunks table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_url TEXT NOT NULL,
                    doc_title TEXT NOT NULL,
                    section_path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    raw_content TEXT,
                    char_count INTEGER,
                    token_estimate INTEGER,
                    chunk_index INTEGER,
                    metadata_json TEXT,
                    FOREIGN KEY(doc_url) REFERENCES documents(url) ON DELETE CASCADE
                )
                """
            )

            # 3. Chunks FTS5 table
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    doc_title,
                    section_path,
                    content,
                    tokenize='porter unicode61'
                )
                """
            )
            conn.commit()

    def save_document_and_chunks(
        self, doc: ScrapedDocument, chunks: List[DocumentChunk]
    ) -> None:
        """Atomically upsert a document and its chunks, keeping FTS in sync."""
        with self._get_connection() as conn:
            # Delete existing document record & chunks if present (triggers cascade)
            conn.execute("DELETE FROM documents WHERE url = ?", (doc.url,))
            conn.execute("DELETE FROM chunks WHERE doc_url = ?", (doc.url,))
            conn.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_url = ?)",
                (doc.url,),
            )

            # Insert document
            conn.execute(
                """
                INSERT INTO documents (
                    url, title, headline, description, markdown_content,
                    raw_text, breadcrumbs, word_count, chunk_count, scraped_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.url,
                    doc.title,
                    doc.headline,
                    doc.description,
                    doc.markdown_content,
                    doc.raw_text,
                    json.dumps(doc.breadcrumbs),
                    doc.word_count,
                    len(chunks),
                    doc.scraped_at,
                    doc.scraped_at,
                ),
            )

            # Insert chunks & FTS entries
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_url, doc_title, section_path,
                        content, raw_content, char_count, token_estimate,
                        chunk_index, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_url,
                        chunk.doc_title,
                        chunk.section_path,
                        chunk.content,
                        chunk.raw_content,
                        chunk.char_count,
                        chunk.token_estimate,
                        chunk.chunk_index,
                        json.dumps(chunk.metadata),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO chunks_fts (chunk_id, doc_title, section_path, content)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_title,
                        chunk.section_path,
                        chunk.content,
                    ),
                )

            conn.commit()

    def list_documents(self) -> List[Dict]:
        """List all stored documents with metadata."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT url, title, headline, description, word_count,
                       chunk_count, scraped_at, updated_at
                FROM documents ORDER BY updated_at DESC
                """
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_document(self, url: str) -> Optional[Dict]:
        """Get single document with markdown content."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM documents WHERE url = ?", (url,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            if data.get("breadcrumbs"):
                data["breadcrumbs"] = json.loads(data["breadcrumbs"])
            return data

    def delete_document(self, url: str) -> bool:
        """Delete document, cascading to chunks and cleaning FTS."""
        with self._get_connection() as conn:
            # Clean FTS
            conn.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_url = ?)",
                (url,),
            )
            cursor = conn.execute("DELETE FROM documents WHERE url = ?", (url,))
            conn.commit()
            return cursor.rowcount > 0

    def get_all_chunks(self) -> List[DocumentChunk]:
        """Retrieve all stored chunks across all documents."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM chunks ORDER BY doc_url, chunk_index"
            )
            chunks = []
            for row in cursor.fetchall():
                meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
                chunks.append(
                    DocumentChunk(
                        chunk_id=row["chunk_id"],
                        doc_url=row["doc_url"],
                        doc_title=row["doc_title"],
                        section_path=row["section_path"],
                        content=row["content"],
                        raw_content=row["raw_content"] or "",
                        char_count=row["char_count"] or 0,
                        token_estimate=row["token_estimate"] or 0,
                        chunk_index=row["chunk_index"],
                        metadata=meta,
                    )
                )
            return chunks

    def search_fts(self, query: str, limit: int = 15, use_and: bool = False) -> List[Dict]:
        """
        Query FTS5 full-text index with Okapi BM25 ranking.

        Args:
            query: The expanded search query string.
            limit: Maximum number of results to return.
            use_and: If True, all terms must appear in the chunk (AND mode — higher precision).
                     If False, any term can match (OR mode — higher recall).
                     The hybrid_search() caller should try AND first, then fall back to OR.
        """
        # Sanitize: strip non-word chars, skip pure numbers, skip single-char tokens
        clean_terms = [
            re.sub(r"[^\w\-]", "", w)
            for w in query.split()
            if len(w) >= 2 and not w.isdigit()
        ]
        clean_terms = [t for t in clean_terms if t and len(t) >= 2]
        if not clean_terms:
            return []

        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for t in clean_terms:
            if t not in seen:
                seen.add(t)
                unique_terms.append(t)

        # Build FTS5 MATCH expression
        if use_and and len(unique_terms) > 1:
            # AND mode: all tokens must appear (prefix match on each)
            fts_query = " AND ".join([f'"{term}"*' for term in unique_terms])
        else:
            # OR mode: any token can match
            fts_query = " OR ".join([f'"{term}"*' for term in unique_terms])

        with self._get_connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT c.chunk_id, c.doc_url, c.doc_title, c.section_path,
                           c.content, c.chunk_index,
                           bm25(chunks_fts, 5.0, 3.0, 1.0) AS bm25_rank
                    FROM chunks_fts f
                    JOIN chunks c ON f.chunk_id = c.chunk_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY bm25_rank
                    LIMIT ?
                    """,
                    (fts_query, limit),
                )
                results = []
                for row in cursor.fetchall():
                    d = dict(row)
                    # Convert negative BM25 score to positive relevance score
                    d["score"] = round(1.0 / (1.0 + abs(d["bm25_rank"])), 4)
                    results.append(d)
                return results
            except sqlite3.OperationalError:
                # If AND query fails (too restrictive or syntax error), return empty
                return []
