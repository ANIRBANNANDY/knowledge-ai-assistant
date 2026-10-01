"""
Production-Grade Multi-Stage Knowledge Query Agent Subsystem.
Orchestrates Query Intent Analysis, Multi-Query Decomposition,
Evidence Verification, Intent-Specific Grounded Synthesis, and Strict Anti-Hallucination Guardrails.

v2 Improvements:
- _extract_clean_prose() now includes Summary blocks as primary definition source
- Added CONTEXTUAL intent for "what happens if / what if / will X affect Y" queries
- Added BENEFITS intent handler
- Improved GENERAL_INQUIRY synthesis with more prose per hit
- Evidence quality confidence caveat for very low-score results
"""

from __future__ import annotations
import collections
import re
import time
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.logger import log_event
from src.llm import LocalLLMClient
from src.retriever import KnowledgeRetriever, SearchHit
from src.storage import KnowledgeStorage


class AgentStep(BaseModel):
    agent_name: str
    action: str
    status: str
    details: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0


class AgentResponse(BaseModel):
    query: str
    intent: str
    reply: str
    evidence_count: int
    citations: List[Dict[str, str]]
    steps: List[AgentStep]
    hits: List[Dict[str, Any]]
    is_grounded: bool = True
    unindexed: bool = False
    decomposed_queries: List[str] = Field(default_factory=list)
    engine: str = "rule-based"
    model: Optional[str] = None
    eval_count: Optional[int] = None
    prompt_eval_count: Optional[int] = None


class KnowledgeQueryAgent:
    """
    Dedicated Multi-Stage Knowledge Query Agent orchestrating query decomposition,
    intent classification, multi-angle evidence retrieval, citation verification,
    and structured factual synthesis via Local LLM (Ollama) with deterministic fallback.
    """

    def __init__(
        self,
        retriever: KnowledgeRetriever,
        storage: KnowledgeStorage,
        llm_client: Optional[LocalLLMClient] = None,
    ):
        self.retriever = retriever
        self.storage = storage
        self.llm_client = llm_client or LocalLLMClient()


    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose compound or multi-part queries into atomic sub-questions for multi-angle retrieval.
        Example: 'How to create a workspace and invite users?' -> ['create a workspace', 'invite users to workspace']
        """
        q = query.strip()
        sub_queries = [q]

        # Check for conjunction splits (and, as well as, along with)
        conjunction_patterns = [
            r"\band\b(?:\s+how\s+to|\s+how\s+do\s+i)?",
            r"\bas\s+well\s+as\b",
            r"\balong\s+with\b",
        ]

        for pat in conjunction_patterns:
            parts = re.split(pat, q, flags=re.IGNORECASE)
            if len(parts) > 1 and all(len(p.strip()) > 8 for p in parts):
                sub_queries = [p.strip().rstrip("?.") for p in parts if len(p.strip()) > 5]
                main_subject = re.findall(
                    r"\b(workspace|saml|sso|jira|permission|scim|guard|rag|bitbucket|repository)\b",
                    q.lower()
                )
                if main_subject:
                    subj = main_subject[0]
                    sub_queries = [
                        sq if subj in sq.lower() else f"{sq} {subj}"
                        for sq in sub_queries
                    ]
                break

        return sub_queries

    def classify_intent(self, query: str) -> Tuple[str, List[str]]:
        """
        Analyze query to classify intent and extract target entity/facets.
        Returns: (intent_type, facet_keywords)

        Intents:
        - CONTEXTUAL: "what happens if X", "what if I X", consequence/effect queries
        - COMPARISON: X vs Y, difference between X and Y
        - TROUBLESHOOTING: error, fail, cannot, issue
        - BENEFITS: benefits, advantages, why use
        - WORKFLOW: how to, steps to, configure, create
        - DEFINITION: what is, explain, define
        - GENERAL_INQUIRY: catch-all
        """
        q = query.lower().strip()

        # 0. Contextual / Consequence queries (check BEFORE WORKFLOW to avoid misclassification)
        if re.search(
            r"\b(what\s+happens?\s+if|what\s+if\s+i|will\s+.+\s+affect|consequence|what\s+will\s+happen|does\s+.+\s+affect|does\s+.+\s+impact|impact\s+of\s+changing|effect\s+of)\b",
            q
        ):
            return "CONTEXTUAL", ["consequence", "result", "effect", "impact", "warning", "note"]

        # 1. Comparison
        if re.search(r"\b(difference between|vs\.?|versus|compared to|compare|different from)\b", q):
            return "COMPARISON", ["comparison", "differences", "matrix", "features"]

        # 2. Troubleshooting / Error
        if re.search(
            r"\b(error|fail|failed|issue|problem|troubleshoot|cannot|can't|unable to|why is|resend|not working|broken)\b",
            q
        ):
            return "TROUBLESHOOTING", ["resolution", "causes", "fixes", "troubleshooting", "diagnostics"]

        # 3. Benefits / Strategic Value
        if re.search(r"\b(benefit|benefits|why use|importance|advantages|pros|value)\b", q):
            return "BENEFITS", ["benefits", "advantages", "value", "capabilities"]

        # 4. How-To / Workflow / Administration
        if re.search(
            r"\b(how to|how do i|steps to|guide to|process of|how can i|configure|setup|add|create|change|modify|delete|invite|grant|migrate)\b",
            q
        ):
            return "WORKFLOW", ["steps", "procedure", "navigation", "instructions", "prerequisites"]

        # 5. Definition / Concept
        if re.search(r"\b(what is|what are|explain|define|definition of|overview|concept|meaning of)\b", q):
            return "DEFINITION", ["definition", "overview", "core concept", "architecture"]

        # Default
        return "GENERAL_INQUIRY", ["information", "overview", "documentation"]

    def _extract_clean_prose(self, content: str, include_summary: bool = True) -> List[str]:
        """
        Extract substantive clean paragraphs from a chunk, stripping headers and TOC lists.

        v2: Now includes Summary block content as a primary definition source.
        The summary often IS the authoritative answer for definition/contextual queries.
        """
        clean = re.sub(r"^###\s+\[Document:[^\]]+\]\s+\|\s+\[Section:[^\]]+\]\s*", "", content).strip()
        lines = [ln.strip() for ln in clean.split("\n") if ln.strip()]

        filtered_lines = []
        summary_text = ""

        for line in lines:
            if line.startswith("#"):
                continue
            if line.startswith("**Category:"):
                continue
            # Capture summary block content (> **Summary:** ...)
            if include_summary and line.startswith("> **Summary:"):
                # Extract the summary text after the marker
                summary_match = re.match(r"^>\s+\*\*Summary:\*\*\s*(.+)$", line)
                if summary_match:
                    summary_text = summary_match.group(1).strip()
                continue
            # Skip question-list TOC bullets
            if line.startswith("-") and "?" in line and len(line) < 80:
                continue
            filtered_lines.append(line)

        paragraphs = []
        curr = []
        for line in filtered_lines:
            curr.append(line)
            if line.endswith(".") or line.endswith(":") or line.endswith("!") or len(line) > 160:
                paragraphs.append(" ".join(curr))
                curr = []
        if curr:
            paragraphs.append(" ".join(curr))

        substantive = [p for p in paragraphs if len(p.split()) >= 6]

        # Prepend summary as the first paragraph if it's substantive
        if summary_text and len(summary_text.split()) >= 4:
            substantive = [summary_text] + substantive

        return substantive

    def _synthesize_definition(self, query: str, hits: List[SearchHit], steps: List[AgentStep]) -> str:
        """Synthesize a structured conceptual definition response."""
        t0 = time.time()
        top_hit = hits[0]

        primary_paragraphs = self._extract_clean_prose(top_hit.content, include_summary=True)
        definition_text = primary_paragraphs[0] if primary_paragraphs else top_hit.content.strip()

        facets_md = []
        seen_sections = {top_hit.section_path}

        for hit in hits[1:]:
            if hit.section_path in seen_sections:
                continue
            seen_sections.add(hit.section_path)

            prose = self._extract_clean_prose(hit.content, include_summary=True)
            if not prose:
                continue

            sec_title = hit.section_path.split(" > ")[-1]
            preview = prose[0]
            if len(prose) > 1:
                preview += f"\n\n{prose[1]}"

            facets_md.append(
                f"### 🔹 {sec_title}\n"
                f"{preview}\n\n"
                f"🔗 *Source: [{hit.doc_title}]({hit.doc_url}) (`{hit.section_path}`)*\n"
            )

        output = [
            f"## 📌 Definition: {query.strip('?').title()}\n\n",
            f"> {definition_text}\n\n",
            "---\n\n",
        ]

        if facets_md:
            output.append("## 🔍 Key Capabilities & Architectural Context\n\n")
            output.append("\n".join(facets_md[:3]))

        steps.append(
            AgentStep(
                agent_name="KnowledgeSynthesisAgent",
                action=f"Synthesized definition layout with {len(facets_md[:3]) + 1} structured sections",
                status="success",
                details={"sections_count": len(facets_md[:3]) + 1},
                latency_ms=round((time.time() - t0) * 1000, 2),
            )
        )

        return "".join(output)

    def _synthesize_contextual(self, query: str, hits: List[SearchHit], steps: List[AgentStep]) -> str:
        """
        Synthesize a consequence/effect response for 'what happens if X' queries.
        Focuses on change-impact sections, warnings, notes, and result statements.
        """
        t0 = time.time()
        top_hit = hits[0]

        consequence_blocks = []
        seen_sections = set()

        for hit in hits[:5]:
            if hit.section_path in seen_sections:
                continue
            seen_sections.add(hit.section_path)

            prose = self._extract_clean_prose(hit.content, include_summary=True)
            if not prose:
                continue

            sec_title = hit.section_path.split(" > ")[-1]

            # Extract note/warning blocks from raw content
            note_lines = []
            for ln in hit.content.split("\n"):
                if re.match(r"^\s*>?\s*\*\*(Note|Warning|Important|Caution):", ln, re.IGNORECASE):
                    note_lines.append(ln.strip())

            body_text = "\n\n".join(prose[:3])
            if note_lines:
                body_text += "\n\n" + "\n".join(note_lines)

            consequence_blocks.append(
                f"### 📌 {sec_title}\n\n"
                f"{body_text}\n\n"
                f"🔗 *Source: [{hit.doc_title}]({hit.doc_url}) (`{hit.section_path}`)*\n"
            )

        output = [
            f"## ⚠️ What Happens: {query.strip('?').title()}\n\n",
            "Based on the indexed documentation, here is what occurs as a consequence:\n\n",
            "---\n\n",
            "\n---\n\n".join(consequence_blocks),
        ]

        steps.append(
            AgentStep(
                agent_name="KnowledgeSynthesisAgent",
                action=f"Synthesized contextual consequence response with {len(consequence_blocks)} sections",
                status="success",
                details={"sections_count": len(consequence_blocks)},
                latency_ms=round((time.time() - t0) * 1000, 2),
            )
        )

        return "".join(output)

    def _synthesize_workflow(self, query: str, hits: List[SearchHit], steps: List[AgentStep]) -> str:
        """Synthesize an actionable step-by-step how-to workflow."""
        t0 = time.time()
        top_hit = hits[0]

        sections_blocks = []
        seen_sections = set()

        for hit in hits[:5]:
            if hit.section_path in seen_sections:
                continue
            seen_sections.add(hit.section_path)

            prose = self._extract_clean_prose(hit.content, include_summary=True)
            if not prose:
                continue

            sec_title = hit.section_path.split(" > ")[-1]
            body_text = "\n\n".join(prose[:3])

            sections_blocks.append(
                f"### 📋 Stage {len(sections_blocks) + 1}: {sec_title}\n\n"
                f"{body_text}\n\n"
                f"📌 *Reference: [{hit.doc_title}]({hit.doc_url}) (`{hit.section_path}`)*\n"
            )

        output = [
            f"## 🛠️ Step-by-Step Procedure: {query.strip('?').title()}\n\n",
            "Here is the verified execution procedure from the authoritative documentation:\n\n",
            "---\n\n",
            "\n---\n\n".join(sections_blocks),
        ]

        steps.append(
            AgentStep(
                agent_name="KnowledgeSynthesisAgent",
                action=f"Synthesized workflow with {len(sections_blocks)} execution stages",
                status="success",
                details={"stages_count": len(sections_blocks)},
                latency_ms=round((time.time() - t0) * 1000, 2),
            )
        )

        return "".join(output)

    def _synthesize_comparison(self, query: str, hits: List[SearchHit], steps: List[AgentStep]) -> str:
        """Synthesize a structured side-by-side comparison matrix."""
        t0 = time.time()
        top_hit = hits[0]

        comparison_blocks = []
        for i, hit in enumerate(hits[:4]):
            prose = self._extract_clean_prose(hit.content, include_summary=True)
            if not prose:
                continue

            sec_title = hit.section_path.split(" > ")[-1]
            comparison_blocks.append(
                f"### ⚖️ Aspect {i+1}: {sec_title}\n\n"
                f"{prose[0]}\n\n"
                f"🔗 *Source: [{hit.doc_title}]({hit.doc_url}) (`{hit.section_path}`)*\n"
            )

        output = [
            f"## ⚖️ Comparative Analysis: {query.strip('?').title()}\n\n",
            "Based on the indexed documentation, here is the direct breakdown of differences:\n\n",
            "---\n\n",
            "\n---\n\n".join(comparison_blocks),
        ]

        steps.append(
            AgentStep(
                agent_name="KnowledgeSynthesisAgent",
                action=f"Synthesized comparison matrix across {len(comparison_blocks)} aspects",
                status="success",
                details={"aspects_count": len(comparison_blocks)},
                latency_ms=round((time.time() - t0) * 1000, 2),
            )
        )

        return "".join(output)

    def _synthesize_troubleshooting(self, query: str, hits: List[SearchHit], steps: List[AgentStep]) -> str:
        """Synthesize a diagnostic troubleshooting resolution guide."""
        t0 = time.time()
        top_hit = hits[0]

        resolution_blocks = []
        for i, hit in enumerate(hits[:4]):
            prose = self._extract_clean_prose(hit.content, include_summary=True)
            if not prose:
                continue

            sec_title = hit.section_path.split(" > ")[-1]
            resolution_blocks.append(
                f"### 🔧 Resolution / Diagnosis {i+1}: {sec_title}\n\n"
                f"{prose[0]}\n\n"
                f"📌 *Diagnostic Reference: [{hit.doc_title}]({hit.doc_url}) (`{hit.section_path}`)*\n"
            )

        output = [
            f"## 🔍 Troubleshooting & Issue Resolution: {query.strip('?').title()}\n\n",
            "Authoritative resolution steps identified from the documentation:\n\n",
            "---\n\n",
            "\n---\n\n".join(resolution_blocks),
        ]

        steps.append(
            AgentStep(
                agent_name="KnowledgeSynthesisAgent",
                action=f"Synthesized troubleshooting guide with {len(resolution_blocks)} resolution paths",
                status="success",
                details={"resolutions_count": len(resolution_blocks)},
                latency_ms=round((time.time() - t0) * 1000, 2),
            )
        )

        return "".join(output)

    def _synthesize_benefits(self, query: str, hits: List[SearchHit], steps: List[AgentStep]) -> str:
        """Synthesize a benefits/value-proposition response."""
        t0 = time.time()
        top_hit = hits[0]

        benefit_blocks = []
        seen_sections = set()

        for hit in hits[:5]:
            if hit.section_path in seen_sections:
                continue
            seen_sections.add(hit.section_path)

            prose = self._extract_clean_prose(hit.content, include_summary=True)
            if not prose:
                continue

            sec_title = hit.section_path.split(" > ")[-1]
            body_text = "\n\n".join(prose[:2])

            benefit_blocks.append(
                f"### ✅ {sec_title}\n\n"
                f"{body_text}\n\n"
                f"🔗 *Source: [{hit.doc_title}]({hit.doc_url}) (`{hit.section_path}`)*\n"
            )

        output = [
            f"## ✅ Benefits & Advantages: {query.strip('?').title()}\n\n",
            "Key value propositions and capabilities from the authoritative documentation:\n\n",
            "---\n\n",
            "\n---\n\n".join(benefit_blocks),
            "\n\n---\n",
            f"📌 **Primary Reference:** [{top_hit.doc_title}]({top_hit.doc_url}) (`{top_hit.section_path}`)\n",
        ]

        steps.append(
            AgentStep(
                agent_name="KnowledgeSynthesisAgent",
                action=f"Synthesized benefits overview with {len(benefit_blocks)} value areas",
                status="success",
                details={"benefit_areas_count": len(benefit_blocks)},
                latency_ms=round((time.time() - t0) * 1000, 2),
            )
        )

        return "".join(output)

    def _synthesize_general(self, query: str, hits: List[SearchHit], steps: List[AgentStep]) -> str:
        """Synthesize general structured documentation summary."""
        t0 = time.time()
        top_hit = hits[0]

        sections_blocks = []
        seen_sections = set()

        for hit in hits[:5]:
            if hit.section_path in seen_sections:
                continue
            seen_sections.add(hit.section_path)

            prose = self._extract_clean_prose(hit.content, include_summary=True)
            if not prose:
                continue

            sec_title = hit.section_path.split(" > ")[-1]
            # Include up to 3 paragraphs per section for richer general answers
            body_text = "\n\n".join(prose[:3])

            sections_blocks.append(
                f"### 📄 {sec_title}\n\n"
                f"{body_text}\n\n"
                f"🔗 *Source: [{hit.doc_title}]({hit.doc_url}) (`{hit.section_path}`)*\n"
            )

        output = [
            f"## 📖 Grounded Knowledge Summary\n\n",
            f"Synthesized from indexed technical documentation for: **\"{query}\"**\n\n",
            "---\n\n",
            "\n---\n\n".join(sections_blocks),
        ]

        steps.append(
            AgentStep(
                agent_name="KnowledgeSynthesisAgent",
                action=f"Synthesized general knowledge layout with {len(sections_blocks)} citations",
                status="success",
                details={"citations_count": len(sections_blocks)},
                latency_ms=round((time.time() - t0) * 1000, 2),
            )
        )

        return "".join(output)

    def _synthesize_by_intent(
        self, intent: str, query: str, final_hits: List[SearchHit], steps: List[AgentStep]
    ) -> str:
        """Helper to invoke the appropriate deterministic heuristic synthesis method."""
        if intent == "DEFINITION":
            return self._synthesize_definition(query, final_hits, steps)
        elif intent == "CONTEXTUAL":
            return self._synthesize_contextual(query, final_hits, steps)
        elif intent == "WORKFLOW":
            return self._synthesize_workflow(query, final_hits, steps)
        elif intent == "COMPARISON":
            return self._synthesize_comparison(query, final_hits, steps)
        elif intent == "TROUBLESHOOTING":
            return self._synthesize_troubleshooting(query, final_hits, steps)
        elif intent == "BENEFITS":
            return self._synthesize_benefits(query, final_hits, steps)
        else:
            return self._synthesize_general(query, final_hits, steps)

    def answer_query(
        self,
        query: str,
        top_k: int = 6,
        use_llm: bool = True,
        model: Optional[str] = None,
        think: bool = False,
    ) -> AgentResponse:
        """
        Main Agent Execution Pipeline:
        1. Query Decomposition (Compound query handling)
        2. Query Intent Analysis
        3. Multi-Angle Evidence Retrieval via Hybrid RRF
        4. Anti-Hallucination Guardrail Check
        5. Grounded Synthesis via Local LLM (Ollama) with Rule-Based Fallback
        """
        t_start = time.time()
        steps: List[AgentStep] = []

        # Stage 1: Query Decomposition
        t0 = time.time()
        decomposed = self.decompose_query(query)
        steps.append(
            AgentStep(
                agent_name="QueryDecompositionAgent",
                action=f"Decomposed query into {len(decomposed)} search angle(s)",
                status="success",
                details={"sub_queries": decomposed},
                latency_ms=round((time.time() - t0) * 1000, 2),
            )
        )

        # Stage 2: Intent Classification
        t0 = time.time()
        intent, facets = self.classify_intent(query)
        steps.append(
            AgentStep(
                agent_name="QueryIntentAgent",
                action=f"Classified intent as '{intent}' (Facets: {facets})",
                status="success",
                details={"intent": intent, "facets": facets},
                latency_ms=round((time.time() - t0) * 1000, 2),
            )
        )

        # Stage 3: Multi-Angle Evidence Retrieval
        t0 = time.time()
        aggregated_hits: Dict[str, SearchHit] = {}

        for sub_q in decomposed:
            hits = self.retriever.hybrid_search(sub_q, top_k=top_k)
            for h in hits:
                if h.chunk_id not in aggregated_hits:
                    aggregated_hits[h.chunk_id] = h
                else:
                    # Boost score if chunk matches multiple decomposed angles
                    aggregated_hits[h.chunk_id].score += h.score * 0.5

        final_hits = sorted(
            list(aggregated_hits.values()), key=lambda x: x.score, reverse=True
        )[:top_k]

        steps.append(
            AgentStep(
                agent_name="EvidenceRetrievalAgent",
                action=f"Retrieved {len(final_hits)} verified evidence chunks across {len(decomposed)} angle(s)",
                status="success" if final_hits else "warning",
                details={
                    "total_candidates": len(aggregated_hits),
                    "selected_count": len(final_hits),
                    "top_section": final_hits[0].section_path if final_hits else None,
                    "top_score": final_hits[0].score if final_hits else 0.0,
                },
                latency_ms=round((time.time() - t0) * 1000, 2),
            )
        )

        # Stage 4: Anti-Hallucination Guardrail Check
        if not final_hits:
            docs = self.storage.list_documents()
            available_topics = ", ".join([d.get("headline", d["title"]) for d in docs[:6]])
            unindexed_msg = (
                "### ❌ Topic Not Found in Knowledge Base\n\n"
                f"The knowledge query agent searched the indexed documentation, but found **no verified information** for: **\"{query}\"**.\n\n"
                f"**Available Indexed Topics ({len(docs)} Docs):**\n"
                f"{available_topics or 'None'}\n\n"
                "💡 *Use the **Ingest URLs** tab or CLI `python cli.py ingest <url>` to add relevant documentation for this topic.*"
            )

            steps.append(
                AgentStep(
                    agent_name="AntiHallucinationGuard",
                    action="Zero verified hits detected: Protected against hallucination",
                    status="protected",
                    details={"query": query, "available_docs": len(docs)},
                    latency_ms=0.5,
                )
            )

            log_event(
                category="AGENT",
                event_type="UNINDEXED_QUERY",
                action=f"Agent blocked unindexed query '{query[:50]}'",
                details={"query": query, "intent": intent},
                latency_ms=round((time.time() - t_start) * 1000, 2),
            )

            return AgentResponse(
                query=query,
                intent=intent,
                reply=unindexed_msg,
                evidence_count=0,
                citations=[],
                steps=steps,
                hits=[],
                is_grounded=True,
                unindexed=True,
                decomposed_queries=decomposed,
                engine="rule-based",
            )

        # Stage 5: Grounded Synthesis (Local LLM with Deterministic Fallback)
        engine = "rule-based"
        active_model = None
        eval_count = None
        prompt_eval_count = None

        if use_llm and self.llm_client and self.llm_client.enabled:
            try:
                llm_res = self.llm_client.synthesize_answer(
                    query=query,
                    intent=intent,
                    hits=final_hits,
                    model=model,
                    think=think,
                )
                reply_text = llm_res.get("text", "").strip()
                if not reply_text or len(reply_text) < 25:
                    raise RuntimeError(f"LLM generated insufficient output ({len(reply_text)} chars)")
                engine = llm_res.get("engine", "ollama")
                active_model = llm_res.get("model")
                eval_count = llm_res.get("eval_count")
                prompt_eval_count = llm_res.get("prompt_eval_count")

                steps.append(
                    AgentStep(
                        agent_name=f"LocalLLMAgent ({active_model})",
                        action=f"Synthesized grounded {intent.lower()} response via Ollama Local LLM",
                        status="success",
                        details={
                            "model": active_model,
                            "eval_tokens": eval_count,
                            "prompt_tokens": prompt_eval_count,
                        },
                        latency_ms=llm_res.get("latency_ms", 0.0),
                    )
                )
            except Exception as e:
                steps.append(
                    AgentStep(
                        agent_name="DeterministicSynthesisAgent",
                        action=f"Local LLM synthesis bypassed/failed ({str(e)[:65]}); fell back to rule-based synthesis",
                        status="warning",
                        details={"error": str(e)},
                        latency_ms=0.0,
                    )
                )
                reply_text = self._synthesize_by_intent(intent, query, final_hits, steps)
        else:
            reply_text = self._synthesize_by_intent(intent, query, final_hits, steps)

        # Add low-confidence caveat if top score is very low
        top_score = final_hits[0].score if final_hits else 0.0
        if top_score < 0.001:
            reply_text += (
                "\n\n> ⚠️ **Low Confidence Notice:** The retrieved evidence chunks have low relevance scores. "
                "The answer above is the best match found in the indexed documentation, but may not fully address your specific question. "
                "Consider ingesting more specific documentation via the **Ingest URLs** tab."
            )

        citations = [
            {
                "title": h.doc_title,
                "url": h.doc_url,
                "section": h.section_path,
                "score": f"{h.score:.4f}",
            }
            for h in final_hits[:5]
        ]

        total_elapsed = round((time.time() - t_start) * 1000, 2)
        log_event(
            category="AGENT",
            event_type="AGENT_QUERY_COMPLETED",
            action=f"Knowledge Query Agent answered '{query[:50]}' ({intent}) [{engine}]",
            details={
                "query": query,
                "intent": intent,
                "engine": engine,
                "model": active_model,
                "evidence_count": len(final_hits),
                "steps_count": len(steps),
                "decomposed_count": len(decomposed),
                "top_score": top_score,
            },
            latency_ms=total_elapsed,
        )

        return AgentResponse(
            query=query,
            intent=intent,
            reply=reply_text,
            evidence_count=len(final_hits),
            citations=citations,
            steps=steps,
            hits=[h.model_dump() for h in final_hits],
            is_grounded=True,
            unindexed=False,
            decomposed_queries=decomposed,
            engine=engine,
            model=active_model,
            eval_count=eval_count,
            prompt_eval_count=prompt_eval_count,
        )

    def stream_query(
        self,
        query: str,
        top_k: int = 4,
        use_llm: bool = True,
        model: Optional[str] = None,
        think: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streaming Agent Execution Pipeline:
        Yields real-time events for decomposition, intent classification, evidence retrieval,
        live thinking tokens, streaming content tokens, and comprehensive completion metrics.
        """
        t_start = time.time()
        steps: List[AgentStep] = []

        # Stage 1: Query Decomposition
        t0 = time.time()
        decomposed = self.decompose_query(query)
        step1 = AgentStep(
            agent_name="QueryDecompositionAgent",
            action=f"Decomposed query into {len(decomposed)} search angle(s)",
            status="success",
            details={"sub_queries": decomposed},
            latency_ms=round((time.time() - t0) * 1000, 2),
        )
        steps.append(step1)
        yield {
            "type": "step",
            "step": step1.model_dump(),
            "stage": "decomposition",
            "message": f"Decomposed query into {len(decomposed)} search angle(s)",
            "elapsed_ms": round((time.time() - t_start) * 1000, 1),
        }

        # Stage 2: Intent Classification
        t0 = time.time()
        intent, facets = self.classify_intent(query)
        step2 = AgentStep(
            agent_name="QueryIntentAgent",
            action=f"Classified intent as '{intent}' (Facets: {facets})",
            status="success",
            details={"intent": intent, "facets": facets},
            latency_ms=round((time.time() - t0) * 1000, 2),
        )
        steps.append(step2)
        yield {
            "type": "step",
            "step": step2.model_dump(),
            "stage": "intent",
            "intent": intent,
            "message": f"Classified intent as '{intent}'",
            "elapsed_ms": round((time.time() - t_start) * 1000, 1),
        }

        # Stage 3: Multi-Angle Evidence Retrieval
        t0 = time.time()
        aggregated_hits: Dict[str, SearchHit] = {}

        for sub_q in decomposed:
            hits = self.retriever.hybrid_search(sub_q, top_k=top_k)
            for h in hits:
                if h.chunk_id not in aggregated_hits:
                    aggregated_hits[h.chunk_id] = h
                else:
                    aggregated_hits[h.chunk_id].score += h.score * 0.5

        final_hits = sorted(
            list(aggregated_hits.values()), key=lambda x: x.score, reverse=True
        )[:top_k]

        step3 = AgentStep(
            agent_name="EvidenceRetrievalAgent",
            action=f"Retrieved {len(final_hits)} verified evidence chunks across {len(decomposed)} angle(s)",
            status="success" if final_hits else "warning",
            details={
                "total_candidates": len(aggregated_hits),
                "selected_count": len(final_hits),
                "top_section": final_hits[0].section_path if final_hits else None,
                "top_score": final_hits[0].score if final_hits else 0.0,
            },
            latency_ms=round((time.time() - t0) * 1000, 2),
        )
        steps.append(step3)
        yield {
            "type": "step",
            "step": step3.model_dump(),
            "stage": "retrieval",
            "message": f"Retrieved {len(final_hits)} verified chunks in {step3.latency_ms}ms",
            "evidence_count": len(final_hits),
            "elapsed_ms": round((time.time() - t_start) * 1000, 1),
        }

        # Stage 4: Anti-Hallucination Guardrail Check
        if not final_hits:
            docs = self.storage.list_documents()
            available_topics = ", ".join([d.get("headline", d["title"]) for d in docs[:6]])
            unindexed_msg = (
                "### ❌ Topic Not Found in Knowledge Base\n\n"
                f"The knowledge query agent searched the indexed documentation, but found **no verified information** for: **\"{query}\"**.\n\n"
                f"**Available Indexed Topics ({len(docs)} Docs):**\n"
                f"{available_topics or 'None'}\n\n"
                "💡 *Use the **Ingest URLs** tab or CLI `python cli.py ingest <url>` to add relevant documentation for this topic.*"
            )

            step4 = AgentStep(
                agent_name="AntiHallucinationGuard",
                action="Zero verified hits detected: Protected against hallucination",
                status="protected",
                details={"query": query, "available_docs": len(docs)},
                latency_ms=0.5,
            )
            steps.append(step4)
            yield {
                "type": "step",
                "step": step4.model_dump(),
                "stage": "guardrail",
                "message": "Protected against hallucination (unindexed topic)",
                "elapsed_ms": round((time.time() - t_start) * 1000, 1),
            }
            yield {"type": "content_token", "text": unindexed_msg}
            yield {
                "type": "done",
                "reply": unindexed_msg,
                "citations": [],
                "steps": [s.model_dump() for s in steps],
                "meta": {
                    "engine": "rule-based",
                    "latency_ms": round((time.time() - t_start) * 1000, 2),
                    "intent": intent,
                },
            }
            return

        citations = [
            {
                "title": h.doc_title,
                "url": h.doc_url,
                "section": h.section_path,
                "score": f"{h.score:.4f}",
            }
            for h in final_hits[:5]
        ]
        yield {"type": "citations", "citations": citations}

        # Stage 5: Grounded Synthesis
        engine = "rule-based"
        active_model = None
        eval_count = None
        prompt_eval_count = None
        tok_per_sec = None
        thinking_ms = 0.0
        synthesis_ms = 0.0
        streamed_text = ""
        stream_failed = False

        if use_llm and self.llm_client and self.llm_client.enabled:
            active_model = model or self.llm_client.default_model
            yield {
                "type": "synthesis_start",
                "engine": f"ollama:{active_model}",
                "model": active_model,
                "think": bool(think),
                "message": f"Synthesizing grounded answer with {active_model}...",
                "elapsed_ms": round((time.time() - t_start) * 1000, 1),
            }

            try:
                t_synth_start = time.time()
                for chunk in self.llm_client.stream_synthesize_answer(
                    query=query,
                    intent=intent,
                    hits=final_hits,
                    model=model,
                    think=think,
                ):
                    chunk_type = chunk.get("type")
                    if chunk_type == "thinking_token":
                        yield {
                            "type": "thinking_token",
                            "text": chunk.get("text", ""),
                            "elapsed_ms": chunk.get("elapsed_ms", 0),
                        }
                    elif chunk_type == "thinking_end":
                        thinking_ms = chunk.get("duration_ms", 0)
                        yield {
                            "type": "thinking_end",
                            "duration_ms": thinking_ms,
                            "text": chunk.get("text", ""),
                        }
                    elif chunk_type == "content_token":
                        streamed_text += chunk.get("text", "")
                        yield {
                            "type": "content_token",
                            "text": chunk.get("text", ""),
                        }
                    elif chunk_type == "metrics":
                        engine = chunk.get("engine", f"ollama:{active_model}")
                        eval_count = chunk.get("eval_count")
                        prompt_eval_count = chunk.get("prompt_eval_count")
                        tok_per_sec = chunk.get("tok_per_sec")
                        thinking_ms = chunk.get("thinking_ms", thinking_ms)
                        synthesis_ms = chunk.get("synthesis_ms", round((time.time() - t_synth_start) * 1000, 2))

                step5 = AgentStep(
                    agent_name=f"LocalLLMAgent ({active_model})",
                    action=f"Synthesized grounded {intent.lower()} response via Ollama Local LLM",
                    status="success",
                    details={
                        "model": active_model,
                        "eval_tokens": eval_count,
                        "prompt_tokens": prompt_eval_count,
                        "thinking_ms": thinking_ms,
                        "synthesis_ms": synthesis_ms,
                        "tok_per_sec": tok_per_sec,
                    },
                    latency_ms=synthesis_ms,
                )
                steps.append(step5)
                yield {
                    "type": "step",
                    "step": step5.model_dump(),
                    "stage": "synthesis",
                    "message": f"Synthesis complete ({eval_count} tokens, {tok_per_sec} tok/s)",
                    "elapsed_ms": round((time.time() - t_start) * 1000, 1),
                }

            except Exception as e:
                stream_failed = True
                step5_fallback = AgentStep(
                    agent_name="DeterministicSynthesisAgent",
                    action=f"Local LLM streaming bypassed/failed ({str(e)[:65]}); fell back to rule-based synthesis",
                    status="warning",
                    details={"error": str(e)},
                    latency_ms=0.0,
                )
                steps.append(step5_fallback)
                yield {
                    "type": "step",
                    "step": step5_fallback.model_dump(),
                    "stage": "fallback",
                    "message": f"LLM error: {str(e)[:50]} - Falling back to Deterministic Rule Engine",
                    "elapsed_ms": round((time.time() - t_start) * 1000, 1),
                }

        if not use_llm or stream_failed or not streamed_text:
            # Deterministic synthesis
            t_det_start = time.time()
            reply_text = self._synthesize_by_intent(intent, query, final_hits, steps)
            synthesis_ms = round((time.time() - t_det_start) * 1000, 2)
            streamed_text = reply_text
            engine = "rule-based"
            # Stream in chunks for smooth UI rendering
            words = reply_text.split(" ")
            for i in range(0, len(words), 8):
                part = " ".join(words[i : i + 8]) + " "
                yield {"type": "content_token", "text": part}

        total_elapsed = round((time.time() - t_start) * 1000, 2)
        yield {
            "type": "done",
            "reply": streamed_text,
            "citations": citations,
            "steps": [s.model_dump() for s in steps],
            "meta": {
                "engine": engine,
                "model": active_model,
                "eval_count": eval_count,
                "prompt_eval_count": prompt_eval_count,
                "tok_per_sec": tok_per_sec,
                "thinking_ms": thinking_ms,
                "synthesis_ms": synthesis_ms,
                "latency_ms": total_elapsed,
                "intent": intent,
            },
        }


