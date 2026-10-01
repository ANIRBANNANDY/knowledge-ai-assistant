"""
High-fidelity web documentation scraper.
Extracts structured Markdown, JSON-LD metadata, headings, tables, panels, and clean text.
Includes unicode normalization and clean entity decoding to prevent hallucination.
"""

from __future__ import annotations
import datetime
import html as html_lib
import json
import os
import re
import time
import unicodedata
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import httpx
from lxml import html, etree

try:
    from curl_cffi import requests as curl_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

from src.logger import log_event


class ScrapedDocument(BaseModel):
    url: str
    title: str
    headline: str
    description: str
    markdown_content: str
    raw_text: str
    breadcrumbs: List[str] = Field(default_factory=list)
    word_count: int = 0
    scraped_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


def clean_unicode(text: str) -> str:
    """Normalize unicode, strip non-breaking spaces and zero-width artifacts."""
    if not text:
        return ""
    # Normalize unicode forms
    norm = unicodedata.normalize("NFKD", text)
    # Replace non-breaking spaces and other irregular whitespace
    norm = norm.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    # Normalize multiple whitespace within lines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in norm.splitlines()]
    return "\n".join(lines)


def parse_pdf_bytes(
    pdf_bytes: bytes,
    source_identifier: str = "document.pdf",
    title_override: Optional[str] = None,
) -> ScrapedDocument:
    """
    High-fidelity PDF document extractor.
    Extracts text, page boundaries, headings, tables, and metadata from PDF bytes.
    Preserves hierarchical breadcrumbs per page for precise RAG citations.
    """
    t0 = time.time()
    title = title_override or ""
    author = ""
    page_texts: List[tuple[int, str]] = []
    total_pages = 0

    # 1. Primary extractor: PyMuPDF (fitz) - ultra-fast & preserves tables
    try:
        import fitz
        doc_fitz = fitz.open(stream=pdf_bytes, filetype="pdf")
        meta = doc_fitz.metadata or {}
        if not title and meta.get("title"):
            title = clean_unicode(meta.get("title"))
        if meta.get("author"):
            author = clean_unicode(meta.get("author"))

        total_pages = len(doc_fitz)

        for idx in range(total_pages):
            page_num = idx + 1
            page = doc_fitz[idx]

            # Extract any structured tables on page
            tables_md: List[str] = []
            try:
                tabs = page.find_tables()
                if tabs and tabs.tables:
                    for t in tabs.tables:
                        df_rows = t.extract()
                        if df_rows and len(df_rows) > 0:
                            headers = [clean_unicode(str(c or "").strip()) for c in df_rows[0]]
                            if any(headers):
                                table_lines = ["| " + " | ".join(headers) + " |"]
                                table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                                for r in df_rows[1:]:
                                    cells = [clean_unicode(str(c or "").strip()).replace("|", "\\|") for c in r]
                                    table_lines.append("| " + " | ".join(cells) + " |")
                                tables_md.append("\n".join(table_lines))
            except Exception:
                pass

            page_text = clean_unicode(page.get_text("text").strip())

            combined_page = []
            if page_text:
                combined_page.append(page_text)
            if tables_md:
                combined_page.extend(tables_md)

            if combined_page:
                page_texts.append((page_num, "\n\n".join(combined_page)))

        doc_fitz.close()
    except Exception as e_fitz:
        # 2. Fallback extractor: pypdf
        try:
            from io import BytesIO
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(pdf_bytes))
            if not title and reader.metadata and reader.metadata.title:
                title = clean_unicode(reader.metadata.title)
            total_pages = len(reader.pages)
            for idx, p in enumerate(reader.pages):
                txt = clean_unicode(p.extract_text() or "").strip()
                if txt:
                    page_texts.append((idx + 1, txt))
        except Exception as e_pypdf:
            raise ValueError(f"Failed to parse PDF bytes: PyMuPDF ({e_fitz}), pypdf fallback ({e_pypdf})")

    # Generate clean human-readable title if missing from metadata
    if not title:
        base_name = os.path.basename(source_identifier.split("?")[0].replace("\\", "/"))
        name_without_ext = re.sub(r"\.pdf$", "", base_name, flags=re.IGNORECASE)
        title = " ".join(re.split(r"[-_]+", name_without_ext)).strip().title()
        if not title:
            title = "PDF Document"

    headline = title
    desc_str = f"PDF document ({total_pages} page(s), {len(page_texts)} indexed)"
    if author:
        desc_str += f" | Author: {author}"

    md_parts = [f"# {headline}\n\n", f"> **Source:** `{source_identifier}` | **Summary:** {desc_str}\n\n"]
    for p_num, text in page_texts:
        md_parts.append(f"## Page {p_num}\n\n{text}\n\n")

    full_markdown = clean_unicode("".join(md_parts))
    raw_text = clean_unicode(" ".join(full_markdown.split()))
    word_count = len(raw_text.split())

    elapsed_ms = (time.time() - t0) * 1000
    log_event(
        category="SCRAPER",
        event_type="PDF_PARSED",
        action=f"Parsed PDF '{headline}' ({total_pages} pages)",
        details={
            "source": source_identifier,
            "pages": total_pages,
            "word_count": word_count,
            "bytes": len(pdf_bytes),
        },
        latency_ms=elapsed_ms,
    )

    return ScrapedDocument(
        url=source_identifier,
        title=title,
        headline=headline,
        description=desc_str,
        markdown_content=full_markdown,
        raw_text=raw_text,
        breadcrumbs=[title, f"Pages 1-{total_pages}"],
        word_count=word_count,
    )


def parse_text_or_markdown(text: str, source_identifier: str = "document.txt") -> ScrapedDocument:
    """Parse plain text or Markdown file into a ScrapedDocument."""
    clean_text_content = clean_unicode(text)
    base_name = os.path.basename(source_identifier.split("?")[0].replace("\\", "/"))
    title = os.path.splitext(base_name)[0].replace("_", " ").replace("-", " ").title()
    if not title:
        title = "Text Document"

    full_md = f"# {title}\n\n> **Source:** `{source_identifier}`\n\n{clean_text_content}"
    raw_text = clean_unicode(" ".join(clean_text_content.split()))
    return ScrapedDocument(
        url=source_identifier,
        title=title,
        headline=title,
        description=f"Uploaded text document ({len(raw_text.split())} words)",
        markdown_content=full_md,
        raw_text=raw_text,
        breadcrumbs=[title],
        word_count=len(raw_text.split()),
    )


class WebScraper:
    def __init__(self, timeout: float = 25.0):
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        }

    def fetch_raw(self, url: str) -> tuple[bytes, str]:
        """
        Fetch raw bytes and content-type using browser impersonation (curl_cffi)
        with graceful fallback to httpx.
        """
        t0 = time.time()
        last_error = None

        # 1. Try curl_cffi with Chrome impersonation
        if HAS_CURL_CFFI:
            try:
                resp = curl_requests.get(
                    url,
                    headers=self.headers,
                    impersonate="chrome124",
                    timeout=self.timeout,
                    allow_redirects=True,
                )
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                elapsed_ms = (time.time() - t0) * 1000
                log_event(
                    category="SCRAPER",
                    event_type="HTTP_FETCH",
                    action=f"Fetched raw {url} via curl_cffi (Chrome impersonated)",
                    details={"status_code": resp.status_code, "bytes": len(resp.content), "content_type": ctype},
                    latency_ms=elapsed_ms,
                )
                return resp.content, ctype
            except Exception as e:
                last_error = e

        # 2. Fallback to standard httpx client
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                resp = client.get(url, headers=self.headers)
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                elapsed_ms = (time.time() - t0) * 1000
                log_event(
                    category="SCRAPER",
                    event_type="HTTP_FETCH",
                    action=f"Fetched raw {url} via httpx",
                    details={"status_code": resp.status_code, "bytes": len(resp.content), "content_type": ctype},
                    latency_ms=elapsed_ms,
                )
                return resp.content, ctype
        except Exception as e:
            if last_error:
                raise last_error from e
            raise e

    def fetch_url(self, url: str) -> str:
        """Fetch HTML/plain text string from a URL."""
        content, _ = self.fetch_raw(url)
        return content.decode("utf-8", errors="replace")

    def parse_html_to_markdown(self, element) -> str:
        """Convert lxml html element to clean markdown representation."""
        if element is None:
            return ""

        # Remove noise elements
        for tag in ["script", "style", "noscript", "svg", "nav", "footer", "header", "iframe", "button"]:
            for el in element.xpath(f".//{tag}"):
                parent = el.getparent()
                if parent is not None:
                    parent.remove(el)

        # Remove sidebars, navigation bars, feedback forms, modals, cookie notices, and table of contents
        for el in element.xpath(
            './/*[contains(@class, "sidebar") or contains(@class, "toc") or contains(@class, "table-of-contents") or contains(@class, "feedback") or contains(@class, "slide-nav") or contains(@class, "slimnav") or contains(@class, "slimheader") or contains(@class, "cookie") or contains(@class, "modal")]'
        ):
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)

        lines: List[str] = []

        def process_node(node, depth: int = 0):
            tag = node.tag if isinstance(node.tag, str) else ""
            tag = tag.lower()

            if not tag:
                return

            if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                level = int(tag[1])
                prefix = "#" * level
                text = clean_unicode(" ".join(node.itertext()).strip())
                if text:
                    lines.append(f"\n{prefix} {text}\n")
                return

            if tag == "p":
                text = clean_unicode(" ".join(node.itertext()).strip())
                if text:
                    lines.append(f"\n{text}\n")
                return

            if tag in ["ul", "ol"]:
                for i, li in enumerate(node.xpath("./li")):
                    li_text = clean_unicode(" ".join(li.itertext()).strip())
                    if li_text:
                        bullet = f"{i+1}." if tag == "ol" else "-"
                        lines.append(f"{bullet} {li_text}")
                lines.append("")
                return

            if tag == "table":
                rows = node.xpath(".//tr")
                if rows:
                    table_md = []
                    header_done = False
                    for tr in rows:
                        cells = tr.xpath("./th | ./td")
                        cell_texts = [
                            clean_unicode(" ".join(c.itertext()).strip()).replace("|", "\\|")
                            for c in cells
                        ]
                        if not cell_texts or not any(cell_texts):
                            continue
                        table_md.append("| " + " | ".join(cell_texts) + " |")
                        if not header_done:
                            table_md.append(
                                "| "
                                + " | ".join(["---"] * len(cell_texts))
                                + " |"
                            )
                            header_done = True
                    if table_md:
                        lines.append("\n" + "\n".join(table_md) + "\n")
                return

            if tag in ["pre", "code"]:
                code_text = clean_unicode("".join(node.itertext()).strip())
                if code_text:
                    if "\n" in code_text or tag == "pre":
                        lines.append(f"\n```\n{code_text}\n```\n")
                    else:
                        lines.append(f"`{code_text}`")
                return

            if tag == "blockquote" or "panel" in node.get("class", "").lower():
                panel_text = clean_unicode(" ".join(node.itertext()).strip())
                if panel_text:
                    lines.append(f"\n> **Note:** {panel_text}\n")
                return

            # For container elements like div, section, article, main
            for child in node:
                process_node(child, depth + 1)

        process_node(element)
        raw_md = "\n".join(lines)
        # Clean up consecutive empty lines
        cleaned_md = re.sub(r"\n{3,}", "\n\n", raw_md).strip()
        return cleaned_md

    def scrape(self, url: str) -> ScrapedDocument:
        """
        Fetch and parse documentation from a given URL or file path.
        Seamlessly supports HTML documentation pages, remote PDF URLs, and local files.
        """
        t0 = time.time()
        url = url.strip()

        # 1. Handle local file paths (direct path or file:// URI)
        local_path = url[7:] if url.startswith("file://") else url
        if os.path.isfile(local_path):
            with open(local_path, "rb") as f:
                content_bytes = f.read()
            if local_path.lower().endswith(".pdf") or content_bytes.startswith(b"%PDF-"):
                return parse_pdf_bytes(content_bytes, source_identifier=url)
            else:
                return parse_text_or_markdown(content_bytes.decode("utf-8", errors="replace"), source_identifier=url)

        # 2. Remote URL: fetch raw bytes and inspect content-type
        content_bytes, ctype = self.fetch_raw(url)

        # 3. Check if remote content is PDF
        if url.lower().split("?")[0].endswith(".pdf") or "application/pdf" in ctype.lower() or content_bytes.startswith(b"%PDF-"):
            return parse_pdf_bytes(content_bytes, source_identifier=url)

        # 4. Parse as HTML
        html_content = content_bytes.decode("utf-8", errors="replace")
        tree = html.fromstring(html_content)

        # 1. Extract metadata from Title & Meta tags
        title = tree.findtext(".//title") or ""
        title = clean_unicode(" ".join(title.strip().split()))

        description = ""
        meta_desc = tree.xpath('//meta[@name="description"]/@content | //meta[@property="og:description"]/@content')
        if meta_desc:
            description = clean_unicode(str(meta_desc[0]).strip())

        headline = ""
        breadcrumbs: List[str] = []
        json_ld_article_body = ""

        # 2. Extract JSON-LD Schema
        for script in tree.xpath('//script[@type="application/ld+json"]/text()'):
            try:
                data = json.loads(script)
                items = data.get("@graph", [data]) if isinstance(data, dict) else (data if isinstance(data, list) else [data])
                for item in items:
                    if isinstance(item, dict):
                        itype = item.get("@type", "")
                        if itype in ["TechArticle", "Article", "NewsArticle", "WebPage", "HelpArticle"]:
                            json_ld_article_body = clean_unicode(item.get("articleBody", "") or json_ld_article_body)
                            headline = clean_unicode(item.get("headline", "") or item.get("name", "") or headline)
                            if not description and item.get("description"):
                                description = clean_unicode(item.get("description"))
                        elif itype == "BreadcrumbList":
                            element_list = item.get("itemListElement", [])
                            for el in sorted(element_list, key=lambda x: x.get("position", 0)):
                                item_obj = el.get("item", {})
                                if isinstance(item_obj, dict) and item_obj.get("name"):
                                    breadcrumbs.append(clean_unicode(item_obj["name"]))
                                elif isinstance(el.get("name"), str):
                                    breadcrumbs.append(clean_unicode(el["name"]))
            except Exception:
                pass

        if not headline:
            h1_nodes = tree.xpath("//h1//text()")
            if h1_nodes:
                headline = clean_unicode(" ".join(" ".join(h1_nodes).split()))
            else:
                headline = title.split("|")[0].strip() if "|" in title else title

        # 3. Locate Main Content Container by priority
        main_candidates = tree.xpath(
            '//article[@id="maincontent"] | //div[contains(@class, "ak-renderer-document")] | //div[contains(@class, "topic__body")] | //main | //article | //*[@role="main"] | //div[@id="main-content"] | //div[@id="content"]'
        )
        main_element = main_candidates[0] if main_candidates else tree.xpath("//body")[0]

        markdown_body = self.parse_html_to_markdown(main_element)

        # If DOM markdown extraction was sparse but JSON-LD articleBody is rich, enrich it
        if len(markdown_body) < 300 and json_ld_article_body:
            markdown_body = f"{json_ld_article_body}"

        # Combine title and breadcrumbs into document markdown header
        doc_header = f"# {headline}\n\n"
        if description:
            doc_header += f"> **Summary:** {description}\n\n"
        if breadcrumbs:
            doc_header += f"**Category:** {' > '.join(breadcrumbs)}\n\n"

        full_markdown = doc_header + markdown_body

        # Clean non-breaking characters from markdown
        full_markdown = clean_unicode(full_markdown)

        # Raw clean text for full-text indexing
        raw_text = clean_unicode(" ".join(" ".join(full_markdown.split("\n")).split()))
        word_count = len(raw_text.split())

        elapsed_ms = (time.time() - t0) * 1000
        log_event(
            category="SCRAPER",
            event_type="DOC_PARSED",
            action=f"Parsed '{headline}'",
            details={
                "url": url,
                "word_count": word_count,
                "breadcrumbs": breadcrumbs,
                "markdown_length": len(full_markdown),
            },
            latency_ms=elapsed_ms,
        )

        return ScrapedDocument(
            url=url,
            title=title or headline,
            headline=headline or title,
            description=description,
            markdown_content=full_markdown,
            raw_text=raw_text,
            breadcrumbs=breadcrumbs,
            word_count=word_count,
        )
