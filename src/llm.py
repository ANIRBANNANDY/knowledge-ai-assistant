"""
Local LLM Integration Module for Knowledge AI Assistant.
Interfaces with local LLMs running via Ollama (e.g., gemma4:12b, qwen2.5vl:7b, deepseek-r1:7b)
with strict zero-hallucination prompting, latency tracking, and fallback protection.
"""

from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, Generator, List, Optional
import httpx

from src.logger import log_event
from src.retriever import SearchHit

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:12b")
DEFAULT_OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "120.0"))
DEFAULT_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.1"))
DEFAULT_MAX_PREDICT = int(os.environ.get("OLLAMA_MAX_PREDICT", "1536"))
DEFAULT_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "2048"))




class LocalLLMClient:
    """
    Client for interacting with local LLMs running on Ollama.
    Supports model detection, health checking, streaming tokens, thinking metrics,
    and strict zero-hallucination grounded RAG synthesis.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        default_model: str = DEFAULT_OLLAMA_MODEL,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT,
        temperature: float = DEFAULT_TEMPERATURE,
        num_ctx: int = DEFAULT_NUM_CTX,
        max_predict: int = DEFAULT_MAX_PREDICT,
        enabled: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_predict = max_predict
        self.enabled = enabled


    def check_health(self) -> Dict[str, Any]:
        """
        Check if Ollama server is running and get list of available models.
        """
        if not self.enabled:
            return {
                "available": False,
                "enabled": False,
                "active_model": self.default_model,
                "models": [],
                "error": "Local LLM is disabled by configuration.",
            }

        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                    has_active = any(
                        self.default_model == m or self.default_model in m or m.startswith(self.default_model)
                        for m in models
                    )
                    return {
                        "available": True,
                        "enabled": True,
                        "active_model": self.default_model,
                        "models": models,
                        "has_active_model": has_active,
                        "base_url": self.base_url,
                    }
                else:
                    return {
                        "available": False,
                        "enabled": True,
                        "active_model": self.default_model,
                        "models": [],
                        "error": f"Ollama returned HTTP status {resp.status_code}",
                    }
        except Exception as e:
            return {
                "available": False,
                "enabled": True,
                "active_model": self.default_model,
                "models": [],
                "error": f"Could not connect to Ollama at {self.base_url}: {str(e)}",
            }

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List detailed models installed in Ollama.
        """
        try:
            with httpx.Client(timeout=4.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("models", [])
        except Exception:
            pass
        return []

    def set_active_model(self, model_name: str) -> bool:
        """
        Update default model.
        """
        if model_name:
            self.default_model = model_name.strip()
            return True
        return False

    def build_system_prompt(self, intent: str, think: bool = False) -> str:
        """
        Build specialized, strict Zero-Hallucination system prompt conforming to
        Production-Grade Knowledge Base Assistant rules.
        """
        intent_guidelines = {
            "DEFINITION": (
                "Intent: CONCEPTUAL & DEFINITION QUERY.\n"
                "- Structure your answer with:\n"
                "  1. **Core Definition Box**: A clear, 1-2 sentence authoritative definition quoted or synthesized directly from context.\n"
                "  2. **Key Capabilities & Architecture**: Structured bullet points or subsections detailing mechanics, purpose, and hierarchy."
            ),
            "WORKFLOW": (
                "Intent: PROCEDURAL & HOW-TO QUERY.\n"
                "- Structure your answer with:\n"
                "  1. **Prerequisites**: Permissions, role, or workspace plan requirements.\n"
                "  2. **Step-by-Step Procedure**: Clear numbered stages with exact UI navigation paths in bold (e.g., **Settings > Workspace settings > ...**).\n"
                "  3. **Important Notes / Warnings**: Callout alerts for critical rules or common gotchas."
            ),
            "CONTEXTUAL": (
                "Intent: CONSEQUENCE / CONTEXTUAL QUERY ('What happens if...').\n"
                "- Structure your answer with:\n"
                "  1. **Direct Impact**: Clearly explain what will happen, change, or break.\n"
                "  2. **Affected Resources**: Details on repositories, URLs, redirection, bookmarks, permissions, or access keys.\n"
                "  3. **Warnings & Precautions**: Explicit alerts about irreversibility or necessary follow-up actions."
            ),
            "COMPARISON": (
                "Intent: COMPARISON QUERY (X vs Y).\n"
                "- Structure your answer with:\n"
                "  1. **Summary Matrix**: Side-by-side comparison table or comparison criteria.\n"
                "  2. **Architectural & Management Differences**: Clear distinction between legacy/direct and centralized management.\n"
                "  3. **Recommendation / Best Practice**: When to use each."
            ),
            "TROUBLESHOOTING": (
                "Intent: TROUBLESHOOTING / ERROR QUERY.\n"
                "- Structure your answer with:\n"
                "  1. **Identified Root Causes**: Why the issue occurs according to documentation.\n"
                "  2. **Actionable Resolution Steps**: Numbered fix procedure.\n"
                "  3. **Verification Checklist**: How to verify the fix succeeded."
            ),
            "BENEFITS": (
                "Intent: BENEFITS & ADVANTAGES QUERY.\n"
                "- Structure your answer with:\n"
                "  1. **Executive Value Summary**: Primary strategic reason for the feature/architecture.\n"
                "  2. **Core Capabilities & Advantages**: Clear bullet points highlighting operational, security, or collaboration benefits."
            ),
        }

        specific_rule = intent_guidelines.get(
            intent,
            "Intent: GENERAL DOCUMENTATION INQUIRY.\n- Provide a comprehensive, well-structured factual summary with clear subsections."
        )

        reasoning_rule = ""
        if think:
            reasoning_rule = (
                "\n### REASONING & THINKING DIRECTIVES:\n"
                "- Keep internal thinking brief, factual, and strictly under 80 words.\n"
                "- Focus solely on identifying relevant facts from context. Do NOT repeat or over-analyze prompt constraints.\n"
                "- Conclude your thinking promptly and provide your complete, authoritative final answer in clean Markdown."
            )

        return (
            "You are the Production-Grade Knowledge Base AI Assistant for technical documentation.\n\n"
            "### STRICT FACTUAL GROUNDING (ZERO HALLUCINATION PROTOCOL):\n"
            "1. Base EVERY statement STRICTLY AND ONLY on the provided verified context chunks.\n"
            "2. DO NOT extrapolate, assume outside training facts, or invent settings, URLs, or UI paths not in the context.\n"
            "3. If the provided context does NOT contain enough details to fully answer, state clearly: "
            "'Based on the indexed documentation, this information is not mentioned.'\n"
            "4. Quote exact UI navigation paths verbatim in bold (e.g. **Settings > Workspace settings > User directory**).\n"
            "5. Do NOT output a separate 'Primary Citation' or 'Citations' section or bullet list at the end of your response. Verified citations are automatically managed and attached as interactive source pills by the UI.\n"
            "6. Be direct, authoritative, and concise. Avoid conversational preambles. Output clean structured markdown.\n\n"
            f"{specific_rule}\n"
            f"{reasoning_rule}\n"
        )

    def _prepare_prompts(
        self, query: str, intent: str, hits: List[SearchHit], think: bool = False
    ) -> tuple[str, str]:
        """
        Prepare concise, high-relevance prompt context to minimize prompt eval latency on GPU.
        """
        selected_hits = hits[:3]
        context_blocks = []
        for i, hit in enumerate(selected_hits):
            # Clean lines and trim excessive length per chunk
            clean_lines = [line.strip() for line in hit.content.splitlines() if line.strip()]
            clean_content = "\n".join(clean_lines)
            if len(clean_content) > 1200:
                clean_content = clean_content[:1200] + "..."
            context_blocks.append(
                f"### [Source #{i+1}] {hit.doc_title}\n"
                f"- **Section:** {hit.section_path}\n"
                f"- **URL:** {hit.doc_url}\n\n"
                f"{clean_content}\n"
            )
        context_md = "\n---\n".join(context_blocks)
        system_prompt = self.build_system_prompt(intent, think=think)
        user_prompt = (
            f"=== VERIFIED DOCUMENTATION CONTEXT ===\n"
            f"{context_md}\n"
            f"=== END CONTEXT ===\n\n"
            f"User Question: {query}\n\n"
            f"Please provide an authoritative, factually grounded response based strictly on the context above (do NOT add any 'Primary Citation' or citations list at the end):"
        )
        return system_prompt, user_prompt

    def synthesize_answer(
        self,
        query: str,
        intent: str,
        hits: List[SearchHit],
        model: Optional[str] = None,
        think: bool = False,
    ) -> Dict[str, Any]:
        """
        Synthesize a grounded answer using the local Ollama LLM (non-streaming).
        Returns dictionary with text, model, latency_ms, thinking_ms, and token metrics.
        Raises RuntimeError if Ollama request fails.
        """
        t0 = time.time()
        target_model = model or self.default_model

        if not hits:
            raise ValueError("No evidence hits provided for LLM synthesis.")

        system_prompt, user_prompt = self._prepare_prompts(query, intent, hits, think=think)

        predict_tokens = max(self.max_predict, 2048) if think else self.max_predict

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "think": bool(think),
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": predict_tokens,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                req_payload = dict(payload)
                resp = client.post(
                    f"{self.base_url}/api/chat",
                    json=req_payload,
                )

                # Graceful handling for models that do not support thinking mode (e.g., qwen2.5vl)
                if resp.status_code == 400 and req_payload.get("think") and "does not support thinking" in resp.text:
                    log_event(
                        category="LLM",
                        event_type="OLLAMA_FALLBACK",
                        action=f"Model '{target_model}' does not support thinking mode; retrying with direct synthesis",
                        details={"model": target_model, "think": think},
                    )
                    req_payload["think"] = False
                    resp = client.post(f"{self.base_url}/api/chat", json=req_payload)

                # Fallback to /api/generate if /api/chat is not supported
                if resp.status_code == 404:
                    gen_payload = {
                        "model": target_model,
                        "system": system_prompt,
                        "prompt": user_prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_ctx": self.num_ctx,
                            "num_predict": predict_tokens,
                        },
                    }
                    resp = client.post(
                        f"{self.base_url}/api/generate",
                        json=gen_payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    answer_text = data.get("response", "").strip()
                    thinking_text = ""
                else:
                    if resp.status_code >= 400:
                        raise RuntimeError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
                    data = resp.json()
                    msg = data.get("message", {})
                    answer_text = msg.get("content", "").strip()
                    thinking_text = msg.get("thinking", "").strip()

            # Clean any trailing Primary Citations/Sources markdown block that LLM may have generated
            if answer_text:
                answer_text = re.sub(
                    r'(?:\r?\n)+\s*#{1,4}\s*(?:Primary\s+Citations?|Verified\s+Citations?|Citations?|Sources?|References?)\b[\s\S]*$',
                    '',
                    answer_text,
                    flags=re.IGNORECASE
                ).strip()
                answer_text = re.sub(
                    r'(?:\r?\n)+[-*_]{3,}\s*(?:\r?\n)+(?:[📌💡🔗]\s*)?\*{0,2}(?:Primary\s+Citation|Primary\s+Reference|Primary\s+Documentation\s+Hub|Support\s+Hub|Diagnostic\s+Reference|Reference)s?:\*{0,2}[\s\S]*$',
                    '',
                    answer_text,
                    flags=re.IGNORECASE
                ).strip()

            # Strict guardrail: empty or truncated responses must trigger fallback to deterministic engine
            # NEVER replace actual answer with thinking trace
            if not answer_text or len(answer_text) < 25:
                raise RuntimeError(
                    f"Local LLM did not emit a final answer ({len(answer_text)} chars, {len(thinking_text)} thinking chars)"
                )

            elapsed_ms = round((time.time() - t0) * 1000, 2)
            eval_count = data.get("eval_count", 0)
            eval_dur_ns = data.get("eval_duration", 0) or 0
            tok_per_sec = round(eval_count / (eval_dur_ns / 1e9), 1) if eval_dur_ns > 0 else 0.0

            log_event(
                category="LLM",
                event_type="OLLAMA_SYNTHESIS",
                action=f"Ollama ({target_model}) synthesized {intent} answer in {elapsed_ms}ms ({tok_per_sec} tok/s)",
                details={
                    "model": target_model,
                    "intent": intent,
                    "evidence_chunks": len(hits),
                    "eval_count": eval_count,
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "tok_per_sec": tok_per_sec,
                },
                latency_ms=elapsed_ms,
            )

            return {
                "text": answer_text,
                "thinking": thinking_text,
                "model": target_model,
                "latency_ms": elapsed_ms,
                "eval_count": eval_count,
                "prompt_eval_count": data.get("prompt_eval_count"),
                "tok_per_sec": tok_per_sec,
                "engine": f"ollama:{target_model}",
            }

        except Exception as e:
            elapsed_ms = round((time.time() - t0) * 1000, 2)
            log_event(
                category="LLM",
                event_type="OLLAMA_ERROR",
                action=f"Ollama ({target_model}) synthesis failed: {str(e)}",
                details={"query": query, "model": target_model, "error": str(e)},
                status="error",
                latency_ms=elapsed_ms,
            )
            raise RuntimeError(f"Local LLM synthesis failed ({target_model}): {str(e)}") from e

    def stream_synthesize_answer(
        self,
        query: str,
        intent: str,
        hits: List[SearchHit],
        model: Optional[str] = None,
        think: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream synthesized grounded answer token-by-token from local Ollama LLM.
        Yields thinking tokens, content tokens, and comprehensive timing/speed metrics.
        """
        t0 = time.time()
        target_model = model or self.default_model

        if not hits:
            raise ValueError("No evidence hits provided for LLM synthesis.")

        system_prompt, user_prompt = self._prepare_prompts(query, intent, hits, think=think)

        predict_tokens = max(self.max_predict, 2048) if think else self.max_predict

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "think": bool(think),
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": predict_tokens,
            },
        }

        yield {
            "type": "llm_start",
            "model": target_model,
            "think": bool(think),
        }

        thinking_text = ""
        content_text = ""
        think_start = time.time()
        think_end = None
        content_start = None
        eval_count = 0
        prompt_eval_count = 0
        eval_dur_ns = 0

        try:
            with httpx.Client(timeout=self.timeout) as client:
                req_payload = dict(payload)
                cm = client.stream("POST", f"{self.base_url}/api/chat", json=req_payload)
                resp = cm.__enter__()

                # Graceful handling: if model does not support thinking, auto-retry with direct streaming
                if resp.status_code == 400 and req_payload.get("think"):
                    err_msg = resp.read().decode("utf-8", errors="replace")
                    if "does not support thinking" in err_msg:
                        cm.__exit__(None, None, None)
                        log_event(
                            category="LLM",
                            event_type="OLLAMA_FALLBACK",
                            action=f"Model '{target_model}' does not support thinking mode; retrying with direct synthesis",
                            details={"model": target_model, "think": think},
                        )
                        req_payload["think"] = False
                        cm = client.stream("POST", f"{self.base_url}/api/chat", json=req_payload)
                        resp = cm.__enter__()

                if resp.status_code >= 400:
                    err_msg = resp.read().decode("utf-8", errors="replace")
                    cm.__exit__(None, None, None)
                    raise RuntimeError(f"Ollama returned HTTP {resp.status_code}: {err_msg}")

                try:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        msg = chunk.get("message", {})
                        th = msg.get("thinking", "")
                        co = msg.get("content", "")

                        if th:
                            thinking_text += th
                            yield {
                                "type": "thinking_token",
                                "text": th,
                                "elapsed_ms": round((time.time() - think_start) * 1000, 1),
                            }

                        if co:
                            if content_start is None:
                                content_start = time.time()
                                if thinking_text and think_end is None:
                                    think_end = content_start
                                    yield {
                                        "type": "thinking_end",
                                        "duration_ms": round((think_end - think_start) * 1000, 1),
                                        "text": thinking_text,
                                    }
                            content_text += co
                            yield {
                                "type": "content_token",
                                "text": co,
                            }

                        if chunk.get("done"):
                            eval_count = chunk.get("eval_count", 0)
                            prompt_eval_count = chunk.get("prompt_eval_count", 0)
                            eval_dur_ns = chunk.get("eval_duration", 0) or 0
                finally:
                    cm.__exit__(None, None, None)

            # If thinking occurred but thinking_end was never sent (e.g. content was slow to start)
            if thinking_text and think_end is None:
                think_end = time.time()
                yield {
                    "type": "thinking_end",
                    "duration_ms": round((think_end - think_start) * 1000, 1),
                    "text": thinking_text,
                }

            # Guardrail: If no real content was generated, raise RuntimeError to trigger deterministic fallback
            # We NEVER yield thinking_text as content!
            if not content_text or len(content_text) < 25:
                raise RuntimeError(
                    f"Local LLM completed thinking ({len(thinking_text)} chars) but failed to emit a final answer."
                )

            total_elapsed_ms = round((time.time() - t0) * 1000, 2)
            think_dur_ms = round(((think_end or time.time()) - think_start) * 1000, 2) if thinking_text else 0.0
            content_dur_ms = round((time.time() - (content_start or t0)) * 1000, 2)
            tok_per_sec = round(eval_count / (eval_dur_ns / 1e9), 1) if eval_dur_ns > 0 else 0.0

            log_event(
                category="LLM",
                event_type="OLLAMA_STREAM_COMPLETED",
                action=f"Ollama ({target_model}) streamed {intent} answer in {total_elapsed_ms}ms ({tok_per_sec} tok/s)",
                details={
                    "model": target_model,
                    "intent": intent,
                    "think": bool(think),
                    "eval_count": eval_count,
                    "thinking_ms": think_dur_ms,
                    "content_ms": content_dur_ms,
                    "tok_per_sec": tok_per_sec,
                },
                latency_ms=total_elapsed_ms,
            )

            yield {
                "type": "metrics",
                "model": target_model,
                "engine": f"ollama:{target_model}",
                "eval_count": eval_count,
                "prompt_eval_count": prompt_eval_count,
                "latency_ms": total_elapsed_ms,
                "thinking_ms": think_dur_ms,
                "synthesis_ms": content_dur_ms,
                "tok_per_sec": tok_per_sec,
            }

        except Exception as e:
            elapsed_ms = round((time.time() - t0) * 1000, 2)
            log_event(
                category="LLM",
                event_type="OLLAMA_STREAM_ERROR",
                action=f"Ollama stream failed ({target_model}): {str(e)}",
                details={"query": query, "model": target_model, "error": str(e)},
                status="error",
                latency_ms=elapsed_ms,
            )
            raise RuntimeError(f"Local LLM stream failed ({target_model}): {str(e)}") from e

