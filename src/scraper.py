"""
High-fidelity web documentation scraper.
Extracts structured Markdown, JSON-LD metadata, headings, tables, panels, and clean text.
Includes unicode normalization and clean entity decoding to prevent hallucination.
"""

from __future__ import annotations
import datetime
import html as html_lib
import json
import re
import time
import unicodedata
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import httpx
from lxml import html, etree

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


class WebScraper:
    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def fetch_url(self, url: str) -> str:
        t0 = time.time()
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            resp = client.get(url, headers=self.headers)
            resp.raise_for_status()
            elapsed_ms = (time.time() - t0) * 1000
            log_event(
                category="SCRAPER",
                event_type="HTTP_FETCH",
                action=f"Fetched {url}",
                details={"status_code": resp.status_code, "bytes": len(resp.content)},
                latency_ms=elapsed_ms,
            )
            return resp.text

    def parse_html_to_markdown(self, element) -> str:
        """Convert lxml html element to clean markdown representation."""
        if element is None:
            return ""

        # Remove noise elements
        for tag in ["script", "style", "noscript", "svg", "nav", "footer", "iframe", "button"]:
            for el in element.xpath(f".//{tag}"):
                parent = el.getparent()
                if parent is not None:
                    parent.remove(el)

        # Remove sidebars, feedback forms, and table of contents
        for el in element.xpath(
            './/*[contains(@class, "sidebar") or contains(@class, "toc") or contains(@class, "table-of-contents") or contains(@class, "feedback")]'
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
        """Fetch and parse documentation from a given URL."""
        t0 = time.time()
        html_content = self.fetch_url(url)
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
            '//article[@id="maincontent"] | //div[contains(@class, "ak-renderer-document")] | //div[contains(@class, "topic__body")] | //article | //main'
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
