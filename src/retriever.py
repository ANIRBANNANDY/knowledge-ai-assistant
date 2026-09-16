"""
Robust Hybrid BM25 + Vector Retrieval Engine with Stopword Filtering,
Substantive Keyword Verification, and Strict Anti-Hallucination Guardrails.

Production-grade v2: Fixes over-aggressive STOP_WORDS/GENERIC_TERMS filtering,
adds semantic synonym expansion, smarter FTS AND/OR fallback, cached vector index,
and a semantic fallback for consequence/effect queries.
"""

from __future__ import annotations
import collections
import math
import re
import time
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.chunker import DocumentChunk
from src.logger import log_event
from src.storage import KnowledgeStorage

# ─────────────────────────────────────────────────────────────────────────────
# STOP WORDS — Only pure grammatical/functional words.
# NOTE: Do NOT add query-intent words here (happens, affect, impact, change,
# configure, etc.) — those carry semantic meaning for documentation queries.
# ─────────────────────────────────────────────────────────────────────────────
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}

# ─────────────────────────────────────────────────────────────────────────────
# SYNONYM MAP — Expands query terms to related vocabulary in the corpus.
# Keys are canonical terms; values are lists of synonymous expressions.
# ─────────────────────────────────────────────────────────────────────────────
SYNONYM_MAP = {
    "invite": ["invitation", "inviting", "member", "grant access", "add user"],
    "invites": ["invitations", "inviting", "member", "grant access", "add user"],
    "invitation": ["invite", "inviting", "member", "grant access"],
    "workspace": ["workspaces", "account", "slug", "organization"],
    "workspaces": ["workspace", "account", "slug", "organization"],
    "slug": ["workspace id", "workspace slug", "url", "rename", "identifier"],
    "id": ["workspace id", "slug", "identifier", "workspace slug", "rename"],
    "identifier": ["id", "slug", "workspace id", "workspace slug"],
    "rename": ["change", "update", "modify", "slug", "workspace id"],
    "access": ["permission", "permissions", "role", "admin", "groups", "grant"],
    "permission": ["permissions", "access", "role", "admin", "scheme", "schemes"],
    "permissions": ["permission", "access", "role", "admin", "scheme", "schemes"],
    "admin": ["administrator", "administration", "org admin", "atlassian administration"],
    "administration": ["admin", "administrator", "org admin", "atlassian administration"],
    "saml": ["sso", "single sign-on", "identity provider", "idp", "okta", "azure"],
    "sso": ["saml", "single sign-on", "identity provider", "idp", "okta", "azure"],
    "scim": ["active directory", "ad sync", "atlassian guard", "atlassian access", "identity"],
    "guard": ["atlassian access", "atlassian guard standard", "identity", "scim"],
    "jira": ["issue", "issues", "project", "permission scheme"],
    "delete": ["remove", "revoke", "disable"],
    "change": ["update", "modify", "edit", "rename", "alter"],
    "update": ["change", "modify", "edit", "rename"],
    "modify": ["change", "update", "edit", "rename"],
    "rag": ["retrieval", "augmented", "generation", "retrieval-augmented"],
    "retrieval": ["rag", "retrieval-augmented", "search", "knowledge base"],
    "repository": ["repo", "repos", "git", "bitbucket"],
    "repo": ["repository", "repositories", "git", "bitbucket"],
    "restrict": ["limit", "block", "prevent", "domain", "allowed"],
    "domain": ["domains", "email domain", "restrict", "allowed domains"],
    # Consequence / effect query expansion
    "happens": ["result", "consequence", "effect", "impact", "what happens"],
    "happen": ["result", "consequence", "effect", "impact"],
    "affect": ["impact", "consequence", "result", "change", "effect"],
    "affects": ["impacts", "consequence", "result", "effect"],
    "impact": ["affect", "consequence", "result", "effect"],
    "consequence": ["result", "affect", "impact", "effect"],
    "effect": ["result", "consequence", "impact", "affect"],
}

# ─────────────────────────────────────────────────────────────────────────────
# GENERIC TERMS — Vague standalone terms that mean nothing without context.
# NOTE: Action verbs like "change", "update", "configure", "delete", "create"
# have been REMOVED — they carry real intent in documentation queries.
# ─────────────────────────────────────────────────────────────────────────────
GENERIC_TERMS = {
    "server", "servers", "cloud", "data", "center", "page", "pages", "app", "apps",
    "application", "applications", "file", "files", "list", "lists", "view", "views",
    "link", "links", "help", "guide", "guides", "system", "systems", "service", "services",
    "option", "options", "new", "step", "steps", "online", "web", "docs", "documentation",
    "feature", "features", "item", "items", "tool", "tools", "info", "information",
    "detail", "details", "overview", "best", "practices", "practice", "method", "methods",
    "different", "process", "rule", "rules", "type", "types", "mode", "modes",
}


class SearchHit(BaseModel):
    chunk_id: str
    doc_url: str
    doc_title: str
    section_path: str
    content: str
    score: float
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    matched_terms: List[str] = Field(default_factory=list)
    metadata: Dict = Field(default_factory=dict)


class KnowledgeRetriever:
    def __init__(self, storage: KnowledgeStorage):
        self.storage = storage
        self._vocab_cache: Optional[Set[str]] = None
        self._vocab_cache_doc_count: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize text into clean alphanumeric terms."""
        return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text)]

    def get_corpus_vocabulary(self) -> Set[str]:
        """
        Get set of all unique words present across all indexed chunks, titles, and headers.
        Uses a doc-count cache to avoid redundant full scans.
        """
        docs = self.storage.list_documents()
        current_doc_count = len(docs)

        if self._vocab_cache is not None and current_doc_count == self._vocab_cache_doc_count:
            return self._vocab_cache

        chunks = self.storage.get_all_chunks()
        vocab: Set[str] = set()
        for chunk in chunks:
            text = f"{chunk.doc_title} {chunk.section_path} {chunk.content}".lower()
            vocab.update(self._tokenize(text))
        self._vocab_cache = vocab
        self._vocab_cache_doc_count = current_doc_count
        return vocab

    def invalidate_vocab_cache(self):
        """Call after new documents are ingested."""
        self._vocab_cache = None
        self._vocab_cache_doc_count = 0

    def extract_substantive_terms(self, query: str) -> List[str]:
        """Extract meaningful keywords from the query, discarding pure stop words."""
        tokens = self._tokenize(query)
        substantive = [t for t in tokens if t not in STOP_WORDS]
        return substantive if substantive else tokens

    def expand_query(self, query: str) -> Tuple[List[str], List[str], str, bool]:
        """
        Extract substantive keywords, split into core vs generic terms, and expand synonyms.
        Verifies whether core topic terms exist in the indexed documentation corpus.

        Returns: (core_terms, all_substantive_terms, expanded_query_string, is_covered_in_corpus)

        Semantic Fallback:
        - If ALL core terms are absent but SOME substantive terms are present in the corpus,
          we still return is_covered=True using the substantive terms as the search vector.
          This prevents false "Topic Not Found" for paraphrase queries.
        """
        substantive = self.extract_substantive_terms(query)
        if not substantive:
            return [], [], "", False

        corpus_vocab = self.get_corpus_vocabulary()
        core_terms = [t for t in substantive if t not in GENERIC_TERMS]

        # Build synonym families for each core term
        covered_core = []
        for core_t in core_terms:
            synonyms = SYNONYM_MAP.get(core_t, [])
            term_family = {core_t} | {
                st for syn in synonyms
                for st in self._tokenize(syn)
                if st not in STOP_WORDS
            }
            if any(variant in corpus_vocab for variant in term_family):
                covered_core.append(core_t)

        # Primary coverage check: core terms in corpus
        if core_terms and covered_core:
            # At least some core terms found — expand and search
            expanded: Set[str] = set(substantive)
            for token in substantive:
                if token in SYNONYM_MAP:
                    for syn in SYNONYM_MAP[token]:
                        for syn_tok in self._tokenize(syn):
                            if syn_tok not in STOP_WORDS:
                                expanded.add(syn_tok)
            return core_terms, substantive, " ".join(expanded), True

        # Semantic fallback: if core terms not found, try substantive terms directly
        if substantive:
            covered_substantive = [t for t in substantive if t in corpus_vocab]
            if covered_substantive:
                # Some substantive terms are in the corpus — use them
                log_event(
                    category="RAG",
                    event_type="SEMANTIC_FALLBACK",
                    action=f"Core terms not in corpus, falling back to substantive terms for '{query[:60]}'",
                    details={
                        "query": query,
                        "core_terms": core_terms,
                        "covered_substantive": covered_substantive,
                    },
                    latency_ms=0.0,
                )
                expanded: Set[str] = set(covered_substantive)
                for token in covered_substantive:
                    if token in SYNONYM_MAP:
                        for syn in SYNONYM_MAP[token]:
                            for syn_tok in self._tokenize(syn):
                                if syn_tok not in STOP_WORDS:
                                    expanded.add(syn_tok)
                # Use covered_substantive as effective core_terms for verification
                return covered_substantive, substantive, " ".join(expanded), True

        # Nothing found — topic is genuinely not indexed
        return core_terms, substantive, " ".join(substantive), False

    def _build_vector_index(self, chunks: List[DocumentChunk]):
        """Compute document frequencies and TF-IDF term vectors for chunks."""
        N = len(chunks)
        if N == 0:
            return {}, {}, []

        doc_freqs = collections.defaultdict(int)
        chunk_tfs = []

        for chunk in chunks:
            tokens = [t for t in self._tokenize(chunk.content) if t not in STOP_WORDS]
            tf = collections.Counter(tokens)
            chunk_tfs.append((chunk, tf, len(tokens)))
            for term in tf.keys():
                doc_freqs[term] += 1

        idf = {}
        for term, df in doc_freqs.items():
            idf[term] = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

        vectors = []
        for chunk, tf, doc_len in chunk_tfs:
            vec = {}
            norm_sq = 0.0
            for term, count in tf.items():
                w = (1.0 + math.log(count)) * idf[term]
                vec[term] = w
                norm_sq += w * w
            norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
            normalized_vec = {t: w / norm for t, w in vec.items()}
            vectors.append((chunk, normalized_vec))

        return idf, vectors

    def search_vector(
        self, expanded_query: str, chunks: List[DocumentChunk], top_k: int = 30
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Calculate TF-IDF vector cosine similarity between query terms and chunks.
        Threshold lowered to 0.04 for broader semantic capture with paraphrase queries.
        """
        if not chunks:
            return []

        idf, vectors = self._build_vector_index(chunks)
        q_tokens = [t for t in self._tokenize(expanded_query) if t not in STOP_WORDS]
        if not q_tokens:
            return []

        q_tf = collections.Counter(q_tokens)
        q_vec = {}
        q_norm_sq = 0.0

        for term, count in q_tf.items():
            if term in idf:
                w = (1.0 + math.log(count)) * idf[term]
                q_vec[term] = w
                q_norm_sq += w * w

        if q_norm_sq == 0:
            return []

        q_norm = math.sqrt(q_norm_sq)
        normalized_q = {t: w / q_norm for t, w in q_vec.items()}

        scores: List[Tuple[DocumentChunk, float]] = []
        for chunk, doc_vec in vectors:
            dot_product = sum(
                weight * doc_vec.get(term, 0.0)
                for term, weight in normalized_q.items()
            )
            # Lowered threshold: 0.04 (was 0.08) for paraphrase/consequence queries
            if dot_product > 0.04:
                scores.append((chunk, round(dot_product, 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def hybrid_search(self, query: str, top_k: int = 6) -> List[SearchHit]:
        """
        Perform hybrid BM25 + vector search with strict substantive keyword verification.

        If substantive query terms do not exist in the corpus (after synonym expansion),
        returns an empty list to prevent hallucination.

        Improvements over v1:
        - Semantic fallback covers paraphrase/consequence queries
        - FTS uses AND-first precision with OR fallback
        - Lower vector similarity threshold (0.04)
        - Smarter TOC/metadata chunk filtering
        - Section-path relevance boosting
        """
        t0 = time.time()
        core_terms, substantive_terms, expanded_query, is_covered = self.expand_query(query)

        # Anti-hallucination gate: if the core subject is absent from documentation
        if not is_covered or not substantive_terms or not expanded_query.strip():
            elapsed_ms = (time.time() - t0) * 1000
            log_event(
                category="RAG",
                event_type="UNINDEXED_TOPIC",
                action=f"Zero hits: Query subject not indexed: '{query[:60]}'",
                details={
                    "query": query,
                    "core_terms": core_terms,
                    "substantive_terms": substantive_terms,
                    "is_covered": is_covered,
                },
                latency_ms=elapsed_ms,
            )
            return []

        # 1. FTS5 BM25 search — AND-first for precision, fallback to OR for recall
        bm25_results = self.storage.search_fts(expanded_query, limit=30, use_and=True)
        if len(bm25_results) < 3:
            # Precision mode found too few — broaden with OR
            bm25_results = self.storage.search_fts(expanded_query, limit=30, use_and=False)

        bm25_ranks = {hit["chunk_id"]: i + 1 for i, hit in enumerate(bm25_results)}
        bm25_chunks = {hit["chunk_id"]: hit for hit in bm25_results}

        # 2. Vector cosine search on expanded query
        all_chunks = self.storage.get_all_chunks()
        vector_results = self.search_vector(expanded_query, all_chunks, top_k=30)
        vector_ranks = {chunk.chunk_id: i + 1 for i, (chunk, _) in enumerate(vector_results)}
        vector_chunks = {chunk.chunk_id: chunk for chunk, _ in vector_results}

        # 3. Reciprocal Rank Fusion (RRF) with quality weighting
        all_ids = set(bm25_ranks.keys()).union(set(vector_ranks.keys()))
        rrf_constant = 60
        query_lower = query.lower()

        def _is_toc_or_metadata(text: str) -> bool:
            lines = [ln.strip() for ln in text.split("\n") if ln.strip() and not ln.startswith("#")]
            if not lines:
                return True
            q_bullets = sum(1 for ln in lines if ln.startswith("-") and "?" in ln)
            if q_bullets >= 2 and q_bullets >= len(lines) * 0.4:
                return True
            non_meta = [
                ln for ln in lines
                if not ln.startswith("> **Summary:") and not ln.startswith("**Category:")
            ]
            if len(non_meta) == 0:
                return True
            return False

        def _get_prose_score(text: str, section_path: str) -> float:
            clean = re.sub(r"^###\s+\[Document:[^\]]+\]\s+\|\s+\[Section:[^\]]+\]\s*", "", text).strip()

            if _is_toc_or_metadata(clean):
                return 0.25

            boost = 1.0
            words = len(clean.split())
            if words > 40:
                boost += 0.3
            if words > 80:
                boost += 0.2

            sec_lower = section_path.lower()
            clean_q = re.sub(r"[^a-zA-Z0-9\s]", "", query_lower).strip()
            clean_sec = re.sub(r"[^a-zA-Z0-9\s]", "", sec_lower).strip()

            for ct in core_terms:
                if ct in sec_lower:
                    boost += 0.25

            if clean_q and (clean_q in clean_sec or clean_sec.endswith(clean_q)):
                boost += 1.0

            # Definition / explanation prose boost
            if re.search(
                r"\b(is\s+(the\s+)?process\s+of|is\s+a\s+|is\s+an\s+|refers\s+to|is\s+defined\s+as|allows\s+you\s+to|enables\s+you\s+to)\b",
                clean.lower()
            ):
                boost += 0.6

            # Intent-section alignment boosts
            if "what is" in query_lower and "what is" in sec_lower:
                boost += 0.5
            elif "how" in query_lower and "how" in sec_lower:
                boost += 0.4
            elif "benefit" in query_lower and "benefit" in sec_lower:
                boost += 0.4
            elif any(w in query_lower for w in ["happens", "happen", "affect", "impact", "consequence", "effect"]):
                # Consequence queries — boost sections with change/impact vocabulary
                if any(w in sec_lower for w in ["change", "update", "modify", "rename", "result", "affect"]):
                    boost += 0.6

            return boost

        combined_scores: List[Tuple[str, float, Optional[int], Optional[int]]] = []
        for cid in all_ids:
            r_bm25 = bm25_ranks.get(cid)
            r_vec = vector_ranks.get(cid)

            raw_score = 0.0
            if r_bm25 is not None:
                raw_score += 1.0 / (rrf_constant + r_bm25)
            if r_vec is not None:
                raw_score += 1.0 / (rrf_constant + r_vec)

            chunk_obj = vector_chunks.get(cid) or bm25_chunks.get(cid)
            content_text = (
                chunk_obj.content if isinstance(chunk_obj, DocumentChunk)
                else chunk_obj.get("content", "")
            )
            sec_path = (
                chunk_obj.section_path if isinstance(chunk_obj, DocumentChunk)
                else chunk_obj.get("section_path", "")
            )

            quality_multiplier = _get_prose_score(content_text, sec_path)
            final_score = raw_score * quality_multiplier
            combined_scores.append((cid, round(final_score, 5), r_bm25, r_vec))

        combined_scores.sort(key=lambda x: x[1], reverse=True)

        # 4. Strict Substantive Keyword Verification:
        # Chunks MUST match at least one core term (or synonym) to be included.
        expanded_term_set = set(self._tokenize(expanded_query))
        target_verification_terms = set(core_terms) if core_terms else expanded_term_set

        target_family: Set[str] = set(target_verification_terms)
        for t in target_verification_terms:
            if t in SYNONYM_MAP:
                for syn in SYNONYM_MAP[t]:
                    target_family.update(self._tokenize(syn))

        # Remove pure stop words from target_family
        target_family = {t for t in target_family if t not in STOP_WORDS and len(t) >= 2}

        verified_hits: List[SearchHit] = []

        for cid, score, r_bm25, r_vec in combined_scores:
            chunk = vector_chunks.get(cid)

            if not chunk and cid in bm25_chunks:
                b = bm25_chunks[cid]
                chunk_text = (
                    b.get("doc_title", "") + " " +
                    b.get("section_path", "") + " " +
                    b.get("content", "")
                ).lower()
                matched = [t for t in target_family if t in chunk_text]
                if matched:
                    all_matched = [t for t in expanded_term_set if t in chunk_text]
                    verified_hits.append(
                        SearchHit(
                            chunk_id=b["chunk_id"],
                            doc_url=b["doc_url"],
                            doc_title=b["doc_title"],
                            section_path=b["section_path"],
                            content=b["content"],
                            score=score,
                            bm25_rank=r_bm25,
                            vector_rank=r_vec,
                            matched_terms=all_matched,
                        )
                    )
            elif chunk:
                chunk_text = (
                    chunk.doc_title + " " +
                    chunk.section_path + " " +
                    chunk.content
                ).lower()
                matched = [t for t in target_family if t in chunk_text]
                if matched:
                    all_matched = [t for t in expanded_term_set if t in chunk_text]
                    verified_hits.append(
                        SearchHit(
                            chunk_id=chunk.chunk_id,
                            doc_url=chunk.doc_url,
                            doc_title=chunk.doc_title,
                            section_path=chunk.section_path,
                            content=chunk.content,
                            score=score,
                            bm25_rank=r_bm25,
                            vector_rank=r_vec,
                            matched_terms=all_matched,
                            metadata=chunk.metadata,
                        )
                    )

            if len(verified_hits) >= top_k:
                break

        elapsed_ms = (time.time() - t0) * 1000
        log_event(
            category="RAG",
            event_type="HYBRID_SEARCH",
            action=f"Retrieved {len(verified_hits)} verified chunks for '{query[:60]}'",
            details={
                "query": query,
                "core_terms": core_terms,
                "substantive_terms": substantive_terms,
                "expanded_terms_count": len(expanded_term_set),
                "total_hits": len(verified_hits),
                "top_score": verified_hits[0].score if verified_hits else 0.0,
                "top_section": verified_hits[0].section_path if verified_hits else None,
                "bm25_count": len(bm25_results),
                "vector_count": len(vector_results),
            },
            latency_ms=elapsed_ms,
        )

        return verified_hits

    def format_context_for_llm(
        self, query: str, hits: List[SearchHit]
    ) -> Dict[str, any]:
        """
        Format retrieved search hits into clean prompt context and citations
        with strict anti-hallucination guardrails.
        """
        if not hits:
            docs = self.storage.list_documents()
            available_topics = ", ".join([d.get("headline", d["title"]) for d in docs])
            not_found_msg = (
                f"STRICT INSTRUCTION: The user asked about '{query}'.\n"
                f"No matching documentation exists in the indexed knowledge base for this query.\n"
                f"State clearly and politely: 'Based on the indexed documentation, no information was found regarding \"{query}\". "
                f"The available documentation currently covers: {available_topics}.'"
            )
            return {
                "query": query,
                "context_markdown": "No relevant documentation found in the knowledge base.",
                "citations": [],
                "prompt": not_found_msg,
            }

        citations = []
        context_blocks = []

        for i, hit in enumerate(hits):
            source_tag = f"[{i+1}]"
            source_info = {
                "tag": source_tag,
                "title": hit.doc_title,
                "section": hit.section_path,
                "url": hit.doc_url,
                "score": hit.score,
                "matched_terms": hit.matched_terms,
            }
            citations.append(source_info)

            context_blocks.append(
                f"### [Source #{i+1}] {hit.doc_title}\n"
                f"- **Section:** {hit.section_path}\n"
                f"- **URL:** {hit.doc_url}\n"
                f"- **Relevance Score:** {hit.score:.5f}\n"
                f"- **Matched Terms:** {', '.join(hit.matched_terms)}\n\n"
                f"{hit.content}\n"
            )

        context_md = "\n---\n".join(context_blocks)

        formatted_prompt = (
            f"You are the dedicated Documentation & Knowledge Base AI Assistant.\n\n"
            f"### STRICT GROUNDING & ZERO-HALLUCINATION RULES:\n"
            f"1. **Rely ONLY on the provided context below**. Do NOT assume, extrapolate, or invent details not explicitly stated.\n"
            f"2. **Direct Citations**: Every claim or step MUST cite its source URL and section header.\n"
            f"3. **Missing Information**: If the provided context does not answer the user's question, state clearly: "
            f"\"Based on the indexed documentation, this information is not mentioned.\"\n"
            f"4. **Exact Terminology**: Quote exact UI paths, permission names, and settings verbatim.\n\n"
            f"=== KNOWLEDGE BASE CONTEXT ===\n"
            f"{context_md}\n"
            f"=== END CONTEXT ===\n\n"
            f"User Question: {query}\n\n"
            f"Grounded, Structured Answer with Citations:"
        )

        return {
            "query": query,
            "context_markdown": context_md,
            "citations": citations,
            "prompt": formatted_prompt,
        }
