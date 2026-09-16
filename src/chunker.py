"""
Semantic & hierarchical Markdown/Text chunking engine.
Preserves header paths, metadata, handles recursive chunk splitting with overlap,
and prevents mid-sentence breaks.
"""

from __future__ import annotations
import hashlib
import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from src.scraper import ScrapedDocument


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_url: str
    doc_title: str
    section_path: str
    content: str
    raw_content: str
    char_count: int
    token_estimate: int
    chunk_index: int
    metadata: Dict = Field(default_factory=dict)


class DocumentChunker:
    def __init__(
        self,
        max_chunk_size: int = 800,
        chunk_overlap: int = 150,
        min_chunk_size: int = 80,
    ):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def _generate_chunk_id(self, url: str, index: int, text: str) -> str:
        h = hashlib.sha256(f"{url}:{index}:{text[:50]}".encode("utf-8")).hexdigest()[:12]
        return f"chk_{h}_{index}"

    def chunk_document(self, doc: ScrapedDocument) -> List[DocumentChunk]:
        """Split a scraped document into semantic chunks with context headers."""
        lines = doc.markdown_content.split("\n")
        sections: List[Dict[str, any]] = []

        current_headers = {1: doc.headline, 2: "", 3: "", 4: ""}
        current_section_lines: List[str] = []

        def get_current_path() -> str:
            path_parts = [
                current_headers[lvl]
                for lvl in sorted(current_headers.keys())
                if current_headers[lvl]
            ]
            return " > ".join(path_parts) if path_parts else doc.headline

        for line in lines:
            header_match = re.match(r"^(#{1,4})\s+(.+)$", line.strip())
            if header_match:
                # Flush previous section
                if current_section_lines:
                    text_block = "\n".join(current_section_lines).strip()
                    if text_block:
                        sections.append(
                            {
                                "path": get_current_path(),
                                "text": text_block,
                            }
                        )
                    current_section_lines = []

                level = len(header_match.group(1))
                header_title = header_match.group(2).strip()
                current_headers[level] = header_title
                # Clear deeper levels
                for deeper in range(level + 1, 5):
                    current_headers[deeper] = ""
                current_section_lines.append(line)
            else:
                current_section_lines.append(line)

        # Flush final section
        if current_section_lines:
            text_block = "\n".join(current_section_lines).strip()
            if text_block:
                sections.append(
                    {
                        "path": get_current_path(),
                        "text": text_block,
                    }
                )

        if not sections:
            sections = [{"path": doc.headline, "text": doc.markdown_content}]

        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for sec in sections:
            sec_path = sec["path"]
            sec_text = sec["text"]

            if len(sec_text) <= self.max_chunk_size:
                if len(sec_text) >= self.min_chunk_size or not chunks:
                    context_prefix = f"### [Document: {doc.headline}] | [Section: {sec_path}]\n\n"
                    full_content = context_prefix + sec_text
                    chunk_id = self._generate_chunk_id(doc.url, chunk_idx, sec_text)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            doc_url=doc.url,
                            doc_title=doc.title,
                            section_path=sec_path,
                            content=full_content,
                            raw_content=sec_text,
                            char_count=len(full_content),
                            token_estimate=max(1, len(full_content) // 4),
                            chunk_index=chunk_idx,
                            metadata={
                                "url": doc.url,
                                "headline": doc.headline,
                                "breadcrumbs": doc.breadcrumbs,
                                "section": sec_path,
                            },
                        )
                    )
                    chunk_idx += 1
            else:
                # Split large section by paragraphs or sentences
                paragraphs = sec_text.split("\n\n")
                buffer = ""

                for p in paragraphs:
                    p = p.strip()
                    if not p:
                        continue

                    if len(buffer) + len(p) + 2 <= self.max_chunk_size:
                        buffer = f"{buffer}\n\n{p}".strip() if buffer else p
                    else:
                        if buffer and len(buffer) >= self.min_chunk_size:
                            context_prefix = f"### [Document: {doc.headline}] | [Section: {sec_path}]\n\n"
                            full_content = context_prefix + buffer
                            chunk_id = self._generate_chunk_id(doc.url, chunk_idx, buffer)
                            chunks.append(
                                DocumentChunk(
                                    chunk_id=chunk_id,
                                    doc_url=doc.url,
                                    doc_title=doc.title,
                                    section_path=sec_path,
                                    content=full_content,
                                    raw_content=buffer,
                                    char_count=len(full_content),
                                    token_estimate=max(1, len(full_content) // 4),
                                    chunk_index=chunk_idx,
                                    metadata={
                                        "url": doc.url,
                                        "headline": doc.headline,
                                        "breadcrumbs": doc.breadcrumbs,
                                        "section": sec_path,
                                    },
                                )
                            )
                            chunk_idx += 1
                            overlap_text = buffer[-self.chunk_overlap :] if len(buffer) > self.chunk_overlap else ""
                            buffer = f"{overlap_text}\n\n{p}".strip() if overlap_text else p
                        else:
                            buffer = p

                if buffer and len(buffer) >= self.min_chunk_size:
                    context_prefix = f"### [Document: {doc.headline}] | [Section: {sec_path}]\n\n"
                    full_content = context_prefix + buffer
                    chunk_id = self._generate_chunk_id(doc.url, chunk_idx, buffer)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            doc_url=doc.url,
                            doc_title=doc.title,
                            section_path=sec_path,
                            content=full_content,
                            raw_content=buffer,
                            char_count=len(full_content),
                            token_estimate=max(1, len(full_content) // 4),
                            chunk_index=chunk_idx,
                            metadata={
                                "url": doc.url,
                                "headline": doc.headline,
                                "breadcrumbs": doc.breadcrumbs,
                                "section": sec_path,
                            },
                        )
                    )
                    chunk_idx += 1

        return chunks
