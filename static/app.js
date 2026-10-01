/**
 * Knowledge AI Assistant - Interactive Web Application
 * Includes Live MCP & Process Activity Log Streaming
 */

const API_BASE = "";

// State
let currentPrompt = "";
let logInterval = null;

// Helper for safe JSON fetching with structured error handling
async function safeFetchJson(url, options = {}) {
    const resp = await fetch(url, options);
    const text = await resp.text();
    let data;
    try {
        data = text ? JSON.parse(text) : {};
    } catch (e) {
        if (!resp.ok) {
            throw new Error(`Server returned HTTP ${resp.status} (${resp.statusText || 'Error'}): ${text.substring(0, 100)}`);
        }
        throw new Error(`Invalid JSON response: ${text.substring(0, 100)}`);
    }
    if (!resp.ok) {
        const errorMsg = data.detail || data.error || data.message || `HTTP ${resp.status} ${resp.statusText}`;
        throw new Error(typeof errorMsg === "string" ? errorMsg : JSON.stringify(errorMsg));
    }
    return data;
}

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initTextSize();
    setupDropZone();
    loadStatus();
    loadSources();
});

// --- Tab Navigation ---
function initTabs() {
    const navButtons = document.querySelectorAll(".nav-item");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const pageTitle = document.getElementById("page-title");
    const pageDesc = document.getElementById("page-desc");

    const titles = {
        chat: {
            title: "Documentation AI Assistant",
            desc: "Ask questions grounded in your local Atlassian Bitbucket Cloud knowledge base"
        },
        search: {
            title: "Hybrid Search & Chunk Explorer",
            desc: "Inspect BM25 keyword rankings, TF-IDF vector similarity, and RRF fused scores"
        },
        sources: {
            title: "Indexed Knowledge Hub",
            desc: "Manage scraped documentation sources, view full content, and monitor chunk counts"
        },
        ingest: {
            title: "Scrape & Ingest URLs",
            desc: "Add new web documentation to your local hybrid knowledge base"
        },
        logs: {
            title: "Real-Time Activity & MCP Logs",
            desc: "Live stream of Gemini LLM interactions, MCP JSON-RPC tool calls, and RAG latencies"
        }
    };

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");

            navButtons.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const activePane = document.getElementById(`pane-${targetTab}`);
            if (activePane) activePane.classList.add("active");

            if (titles[targetTab]) {
                pageTitle.innerText = titles[targetTab].title;
                pageDesc.innerText = titles[targetTab].desc;
            }

            if (targetTab === "sources") loadSources();
            if (targetTab === "logs") {
                loadLogs();
                if (!logInterval) {
                    logInterval = setInterval(loadLogs, 3000);
                }
            } else {
                if (logInterval) {
                    clearInterval(logInterval);
                    logInterval = null;
                }
            }
        });
    });
}

// State
let activeLlmModel = "gemma4:12b";
let isLlmOnline = false;

// --- Reading Text Size Management ---
function initTextSize() {
    let saved = "medium";
    try {
        saved = localStorage.getItem("preferred-text-size") || "medium";
    } catch (e) {}
    setTextSize(saved);
}

function setTextSize(size) {
    const validSizes = ["small", "medium", "large"];
    if (!validSizes.includes(size)) size = "medium";

    document.documentElement.setAttribute("data-text-size", size);
    try {
        localStorage.setItem("preferred-text-size", size);
    } catch (e) {}

    document.querySelectorAll(".size-pill").forEach(btn => {
        btn.classList.toggle("active", btn.getAttribute("data-size") === size);
    });
}

// --- Status Loading ---
async function loadStatus() {
    try {
        const data = await safeFetchJson(`${API_BASE}/api/status`);
        document.getElementById("stat-docs").innerText = data.total_documents || 0;
        document.getElementById("stat-chunks").innerText = data.total_chunks || 0;
        document.getElementById("stat-words").innerText = (data.total_words || 0).toLocaleString();

        if (data.llm) {
            isLlmOnline = data.llm.available;
            activeLlmModel = data.llm.active_model || "gemma4:12b";

            const statLlm = document.getElementById("stat-llm");
            if (statLlm) {
                statLlm.innerText = isLlmOnline ? activeLlmModel : "Offline";
                statLlm.style.color = isLlmOnline ? "var(--success)" : "var(--warning)";
            }

            const currentModelTag = document.getElementById("current-model-tag");
            if (currentModelTag) currentModelTag.innerText = activeLlmModel;

            const chipPulse = document.getElementById("chip-pulse");
            const chipText = document.getElementById("llm-status-text");
            if (chipText) {
                chipText.innerText = isLlmOnline ? `Ollama Ready (${activeLlmModel})` : "Ollama Offline (Using Rule Engine)";
            }
            if (chipPulse) {
                chipPulse.className = isLlmOnline ? "pulse-emerald" : "pulse-amber";
            }

            // Populate header model selector
            const selectEl = document.getElementById("header-model-select");
            if (selectEl && data.llm.models && data.llm.models.length > 0) {
                selectEl.innerHTML = data.llm.models.map(m => `
                    <option value="${escapeHtml(m)}" ${m === activeLlmModel ? "selected" : ""}>
                        ${escapeHtml(m)}
                    </option>
                `).join("");
            }
        }
    } catch (err) {
        console.error("Failed to load status:", err);
    }
}

let deepThinkingEnabled = false;

function handleLlmToggle(checked) {
    const labelText = document.getElementById("toggle-label-text");
    const deepWrap = document.getElementById("thinking-mode-wrap");
    if (labelText) {
        if (checked) {
            labelText.innerHTML = `🤖 Ollama LLM: <strong id="current-model-tag">${escapeHtml(activeLlmModel)}</strong>`;
            if (deepWrap) deepWrap.style.display = "inline-flex";
        } else {
            labelText.innerHTML = `⚡ Rule-Based Deterministic Engine`;
            if (deepWrap) deepWrap.style.display = "none";
        }
    }
}

function handleDeepThinkToggle(checked) {
    deepThinkingEnabled = checked;
    const label = document.getElementById("deep-think-label");
    if (label) {
        label.innerHTML = checked ? "🧠 Deep Reasoning" : "⚡ Turbo Direct";
    }
}

function toggleThinkingBox(msgId) {
    const box = document.getElementById(`think-${msgId}`);
    if (box) {
        box.classList.toggle("collapsed");
    }
}

// --- Citation Processing & Deduplication Helpers ---
function stripTrailingCitations(md) {
    if (!md) return "";
    let cleaned = md;
    // Strip trailing markdown sections like:
    // ### Primary Citations / ## Citations / ### Verified Citations / ### Sources
    // along with all following bullet items
    cleaned = cleaned.replace(
        /(?:\r?\n)+\s*#{1,4}\s*(?:Primary\s+Citations?|Verified\s+Citations?|Citations?|Sources?|References?)\b[\s\S]*$/i,
        ""
    );
    // Strip trailing single-line reference notes like:
    // 📌 **Primary Citation:** ... / 💡 **Support Hub:** ...
    cleaned = cleaned.replace(
        /(?:\r?\n)+[-*_]{3,}\s*(?:\r?\n)+(?:[📌💡🔗]\s*)?\*{0,2}(?:Primary\s+Citation|Primary\s+Reference|Primary\s+Documentation\s+Hub|Support\s+Hub|Diagnostic\s+Reference|Reference)s?:\*{0,2}[\s\S]*$/i,
        ""
    );
    cleaned = cleaned.replace(
        /(?:\r?\n)+(?:[📌💡🔗]\s*)?\*{0,2}(?:Primary\s+Citation|Primary\s+Reference|Primary\s+Documentation\s+Hub|Support\s+Hub|Diagnostic\s+Reference|Reference)s?:\*{0,2}[\s\S]*$/i,
        ""
    );
    return cleaned.trim();
}

function deduplicateCitations(citations) {
    if (!citations || !citations.length) return [];
    const seen = new Set();
    const result = [];
    for (const c of citations) {
        const url = (c.url || "").trim().toLowerCase();
        const sec = (c.section || "").trim().toLowerCase();
        const key = `${url}::${sec}`;
        if (!seen.has(key)) {
            seen.add(key);
            result.push(c);
        }
    }
    return result;
}

function formatCitationLabel(c) {
    const title = (c.title || "Document").trim();
    let sec = (c.section || "").trim();

    // If section starts with or duplicates title, strip redundant prefix
    if (sec.toLowerCase().startsWith(title.toLowerCase())) {
        sec = sec.substring(title.length).replace(/^[\s>:\-–—]+/, "").trim();
    }
    if (sec) {
        return `${title} (${sec})`;
    }
    return title;
}

function renderCitationsContent(citations) {
    const deduped = deduplicateCitations(citations);
    if (!deduped.length) return "";
    return `
        <span class="citations-header">VERIFIED CITATIONS:</span>
        ${deduped.map(c => {
            const label = formatCitationLabel(c);
            const url = c.url || "#";
            const fullTooltip = `${c.title || ''} (${c.section || ''})`.trim();
            return `
                <a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="citation-link" title="${escapeHtml(fullTooltip)}">
                    <span class="citation-icon">🔗</span>
                    <span class="citation-text">${escapeHtml(label)}</span>
                </a>
            `;
        }).join("")}
    `;
}

// --- Real-Time Streaming Chat Assistant ---
async function handleChatSubmit(e) {
    e.preventDefault();
    const input = document.getElementById("chat-input");
    const query = input.value.trim();
    if (!query) return;

    appendMessage("user", query);
    input.value = "";

    const toggleEl = document.getElementById("toggle-llm");
    const useLlm = toggleEl ? toggleEl.checked : true;
    const modelSelect = document.getElementById("header-model-select");
    const model = modelSelect ? modelSelect.value : activeLlmModel;

    // Create live assistant message element
    const container = document.getElementById("chat-messages");
    const msgId = "msg-" + Date.now();
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant-msg";
    msgDiv.id = msgId;

    const initialBadge = useLlm
        ? `<div class="msg-engine-badge" id="engine-badge-${msgId}">
             <span class="engine-icon">🦙</span>
             <span>Synthesizing with <strong>${escapeHtml(model)}</strong> (Ollama)</span>
             <span class="badge-tag">${deepThinkingEnabled ? "Deep Reasoning" : "Turbo Direct"}</span>
             <span class="badge-stopwatch" id="stopwatch-${msgId}">0.0s</span>
           </div>`
        : `<div class="msg-engine-badge rule-based" id="engine-badge-${msgId}">
             <span class="engine-icon">⚡</span>
             <span>Deterministic Rule Engine</span>
             <span class="badge-tag">Grounded Heuristic</span>
             <span class="badge-stopwatch" id="stopwatch-${msgId}">0.0s</span>
           </div>`;

    msgDiv.innerHTML = `
        <div class="avatar">🤖</div>
        <div class="msg-body">
            ${initialBadge}

            <!-- Live Status Progress Strip -->
            <div class="live-status-strip" id="status-strip-${msgId}">
                <span class="live-pulse-dot"></span>
                <span class="live-status-text" id="status-text-${msgId}">Analyzing query & decomposing search angles...</span>
                <span class="live-timer-chip" id="timer-chip-${msgId}">0.0s</span>
            </div>

            <!-- Expandable Live Thinking Process Card -->
            <div class="thinking-box" id="think-${msgId}" style="display: none;">
                <div class="thinking-header" onclick="toggleThinkingBox('${msgId}')">
                    <span class="think-icon">💭</span>
                    <span class="think-title" id="think-title-${msgId}">Thinking Process</span>
                    <span class="think-timer-badge" id="think-timer-${msgId}">0.0s</span>
                    <span class="think-chevron">▾</span>
                </div>
                <div class="thinking-body markdown-thinking" id="think-body-${msgId}"></div>
            </div>

            <!-- Main Grounded Response Content Area -->
            <div class="markdown-content" id="content-${msgId}">
                <span class="streaming-dot-pulse"></span>
            </div>

            <!-- Verified Citations Card -->
            <div class="citations-box" id="citations-${msgId}" style="display: none;"></div>

            <!-- Execution Pipeline Accordion -->
            <div id="steps-${msgId}" style="display: none;"></div>
        </div>
    `;

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;

    // Start live stopwatch timer
    const tStart = performance.now();
    const timerInterval = setInterval(() => {
        const elapsedSec = ((performance.now() - tStart) / 1000).toFixed(1) + "s";
        const swEl = document.getElementById(`stopwatch-${msgId}`);
        const chipEl = document.getElementById(`timer-chip-${msgId}`);
        if (swEl) swEl.innerText = elapsedSec;
        if (chipEl) chipEl.innerText = elapsedSec;
    }, 100);

    let accumulatedContent = "";
    let accumulatedThinking = "";
    let steps = [];
    let thinkingStarted = false;
    let thinkingEnded = false;

    try {
        const response = await fetch(`${API_BASE}/api/chat/stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: query,
                top_k: 4,
                use_llm: useLlm,
                model: model,
                think: deepThinkingEnabled,
            }),
        });

        if (!response.ok) {
            throw new Error(`Server returned HTTP ${response.status}: ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n\n");
            buffer = lines.pop(); // Keep partial line in buffer

            for (const block of lines) {
                const line = block.trim();
                if (!line.startsWith("data: ")) continue;

                let event;
                try {
                    event = JSON.parse(line.substring(6));
                } catch (e) {
                    continue;
                }

                const evType = event.type;

                if (evType === "step") {
                    const statusText = document.getElementById(`status-text-${msgId}`);
                    if (statusText) statusText.innerText = event.message || event.step?.action;
                    if (event.step) steps.push(event.step);
                } else if (evType === "synthesis_start") {
                    const statusText = document.getElementById(`status-text-${msgId}`);
                    if (statusText) {
                        statusText.innerText = event.think
                            ? `Reasoning & synthesizing with ${escapeHtml(event.model)}...`
                            : `Synthesizing grounded response with ${escapeHtml(event.model)}...`;
                    }
                } else if (evType === "thinking_token") {
                    if (!thinkingStarted) {
                        thinkingStarted = true;
                        const thinkBox = document.getElementById(`think-${msgId}`);
                        if (thinkBox) thinkBox.style.display = "block";
                        const statusStrip = document.getElementById(`status-strip-${msgId}`);
                        if (statusStrip) statusStrip.style.display = "none";
                    }
                    accumulatedThinking += event.text;
                    const thinkBody = document.getElementById(`think-body-${msgId}`);
                    if (thinkBody) {
                        thinkBody.innerHTML = (typeof marked !== "undefined") ? marked.parse(accumulatedThinking) : escapeHtml(accumulatedThinking);
                    }
                    const thinkTimer = document.getElementById(`think-timer-${msgId}`);
                    if (thinkTimer && event.elapsed_ms) {
                        thinkTimer.innerText = (event.elapsed_ms / 1000).toFixed(1) + "s";
                    }
                    container.scrollTop = container.scrollHeight;
                } else if (evType === "thinking_end") {
                    thinkingEnded = true;
                    const thinkBox = document.getElementById(`think-${msgId}`);
                    if (thinkBox) {
                        thinkBox.classList.add("completed");
                        thinkBox.classList.add("collapsed");
                    }
                    const thinkTitle = document.getElementById(`think-title-${msgId}`);
                    if (thinkTitle) {
                        const dur = ((event.duration_ms || 0) / 1000).toFixed(1);
                        thinkTitle.innerHTML = `Thought for <strong>${dur}s</strong>`;
                    }
                    const thinkTimer = document.getElementById(`think-timer-${msgId}`);
                    if (thinkTimer) thinkTimer.style.display = "none";
                } else if (evType === "content_token") {
                    // Hide status strip once content streaming begins
                    const statusStrip = document.getElementById(`status-strip-${msgId}`);
                    if (statusStrip) statusStrip.style.display = "none";

                    accumulatedContent += event.text;
                    const contentEl = document.getElementById(`content-${msgId}`);
                    if (contentEl) {
                        const cleanMd = stripTrailingCitations(accumulatedContent);
                        const parsed = (typeof marked !== "undefined") ? marked.parse(cleanMd) : escapeHtml(cleanMd);
                        contentEl.innerHTML = parsed + '<span class="typing-cursor"></span>';
                    }
                    container.scrollTop = container.scrollHeight;
                } else if (evType === "citations") {
                    const citationsBox = document.getElementById(`citations-${msgId}`);
                    if (citationsBox && event.citations && event.citations.length > 0) {
                        const content = renderCitationsContent(event.citations);
                        if (content) {
                            citationsBox.innerHTML = content;
                            citationsBox.style.display = "flex";
                        }
                    }
                } else if (evType === "done") {
                    clearInterval(timerInterval);

                    // Finalize content with trailing citations cleanly stripped
                    const contentEl = document.getElementById(`content-${msgId}`);
                    if (contentEl) {
                        const rawMd = accumulatedContent || event.reply || "";
                        const finalMd = stripTrailingCitations(rawMd);
                        contentEl.innerHTML = (typeof marked !== "undefined") ? marked.parse(finalMd) : escapeHtml(finalMd);
                    }

                    // Render steps accordion if available
                    const finalSteps = event.steps || steps;
                    const stepsBox = document.getElementById(`steps-${msgId}`);
                    if (stepsBox && finalSteps.length > 0) {
                        stepsBox.style.display = "block";
                        stepsBox.innerHTML = `
                            <details class="reasoning-accordion">
                                <summary>
                                    <span class="trace-icon">⚡</span>
                                    <span>Multi-Stage Agent Pipeline (${finalSteps.length} stages)</span>
                                </summary>
                                <div class="trace-content">
                                    <ul class="trace-steps-list">
                                        ${finalSteps.map(s => `
                                            <li class="trace-step-item">
                                                <span class="trace-step-agent">${escapeHtml(s.agent_name)}</span>
                                                <span class="trace-step-action">${escapeHtml(s.action)}</span>
                                                <span class="trace-step-latency">${s.latency_ms ? s.latency_ms.toFixed(1) + 'ms' : ''}</span>
                                            </li>
                                        `).join("")}
                                    </ul>
                                </div>
                            </details>
                        `;
                    }

                    // Finalize Badge with complete Timing & Speed Metrics
                    const meta = event.meta || {};
                    const badgeEl = document.getElementById(`engine-badge-${msgId}`);
                    if (badgeEl) {
                        const totalSec = (meta.latency_ms ? meta.latency_ms / 1000 : (performance.now() - tStart) / 1000).toFixed(1);
                        const synthSec = (meta.synthesis_ms ? meta.synthesis_ms / 1000 : totalSec).toFixed(1);
                        const thinkSec = meta.thinking_ms ? (meta.thinking_ms / 1000).toFixed(1) : "0.0";
                        const tokPerSec = meta.tok_per_sec ? `${meta.tok_per_sec} tok/s` : "";
                        const tokensCount = meta.eval_count ? `${meta.eval_count} tokens` : "";

                        let timingMetricHtml = "";
                        if (parseFloat(thinkSec) > 0) {
                            timingMetricHtml = `<span class="badge-timing">🧠 Thought for ${thinkSec}s • Synthesized in ${synthSec}s ${tokPerSec ? '(' + tokPerSec + ')' : ''}</span>`;
                        } else {
                            timingMetricHtml = `<span class="badge-timing">⚡ Synthesized in ${synthSec}s ${tokPerSec ? '(' + tokPerSec + ')' : ''}</span>`;
                        }

                        if (meta.engine && meta.engine.startsWith("ollama")) {
                            const modelTag = meta.model || model;
                            badgeEl.className = "msg-engine-badge";
                            badgeEl.innerHTML = `
                                <span class="engine-icon">🦙</span>
                                <span>Synthesized by <strong>${escapeHtml(modelTag)}</strong> (Ollama)</span>
                                <span class="badge-tag">Zero Hallucination</span>
                                ${timingMetricHtml}
                                ${tokensCount ? `<span class="badge-tokens">${tokensCount}</span>` : ""}
                            `;
                        } else {
                            badgeEl.className = "msg-engine-badge rule-based";
                            badgeEl.innerHTML = `
                                <span class="engine-icon">⚡</span>
                                <span>Deterministic Rule Engine</span>
                                <span class="badge-tag">Grounded Heuristic</span>
                                ${timingMetricHtml}
                            `;
                        }
                    }

                    container.scrollTop = container.scrollHeight;
                    loadStatus();
                } else if (evType === "error") {
                    clearInterval(timerInterval);
                    const contentEl = document.getElementById(`content-${msgId}`);
                    if (contentEl) {
                        contentEl.innerHTML = `<div class="error-box">⚠️ ${escapeHtml(event.error)}</div>`;
                    }
                }
            }
        }
    } catch (err) {
        clearInterval(timerInterval);
        const statusStrip = document.getElementById(`status-strip-${msgId}`);
        if (statusStrip) statusStrip.style.display = "none";
        const contentEl = document.getElementById(`content-${msgId}`);
        if (contentEl) {
            contentEl.innerHTML = `<div class="error-box">⚠️ Error communicating with knowledge assistant: ${escapeHtml(err.message)}</div>`;
        }
    }
}

function askPreset(question) {
    const input = document.getElementById("chat-input");
    input.value = question;
    document.getElementById("chat-form").dispatchEvent(new Event("submit"));
}

function appendMessage(role, text) {
    const container = document.getElementById("chat-messages");
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role}-msg`;

    const avatar = role === "user" ? "👤" : "🤖";
    msgDiv.innerHTML = `
        <div class="avatar">${avatar}</div>
        <div class="msg-body">${escapeHtml(text)}</div>
    `;
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function appendAssistantMessage(replyMarkdown, citations, steps, meta = {}) {
    const container = document.getElementById("chat-messages");
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant-msg";

    const cleanedMd = stripTrailingCitations(replyMarkdown);
    const parsedHtml = (typeof marked !== "undefined") ? marked.parse(cleanedMd) : escapeHtml(cleanedMd);

    // Engine badge pill
    let engineBadgeHtml = "";
    if (meta.engine && meta.engine.startsWith("ollama")) {
        const modelName = meta.model || activeLlmModel;
        const tokenBadge = meta.eval_count ? `<span class="badge-tokens">${meta.eval_count} tokens</span>` : "";
        engineBadgeHtml = `
            <div class="msg-engine-badge">
                <span class="engine-icon">🦙</span>
                <span>Synthesized by <strong>${escapeHtml(modelName)}</strong> (Ollama)</span>
                <span class="badge-tag">Zero Hallucination</span>
                ${tokenBadge}
            </div>
        `;
    } else {
        engineBadgeHtml = `
            <div class="msg-engine-badge rule-based">
                <span class="engine-icon">⚡</span>
                <span>Deterministic Rule Engine</span>
                <span class="badge-tag">Grounded Heuristic</span>
            </div>
        `;
    }

    // Citations (industry standard bottom pills, deduplicated, 5-6pt text)
    let citationsHtml = "";
    if (citations && citations.length > 0) {
        const inner = renderCitationsContent(citations);
        if (inner) {
            citationsHtml = `
                <div class="citations-box">
                    ${inner}
                </div>
            `;
        }
    }

    // Expandable Multi-Stage Agent Execution Steps
    let stepsHtml = "";
    if (steps && steps.length > 0) {
        stepsHtml = `
            <details class="reasoning-accordion">
                <summary>
                    <span class="trace-icon">⚡</span>
                    <span>Multi-Stage Agent Pipeline (${steps.length} stages)</span>
                </summary>
                <div class="trace-content">
                    <ul class="trace-steps-list">
                        ${steps.map(s => `
                            <li class="trace-step-item">
                                <span class="trace-step-agent">${escapeHtml(s.agent_name)}</span>
                                <span class="trace-step-action">${escapeHtml(s.action)}</span>
                                <span class="trace-step-latency">${s.latency_ms ? s.latency_ms.toFixed(1) + 'ms' : ''}</span>
                            </li>
                        `).join("")}
                    </ul>
                </div>
            </details>
        `;
    }

    msgDiv.innerHTML = `
        <div class="avatar">🤖</div>
        <div class="msg-body">
            ${engineBadgeHtml}
            <div class="markdown-content">${parsedHtml}</div>
            ${citationsHtml}
            ${stepsHtml}
        </div>
    `;
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function appendLoadingMessage(text = "Searching local knowledge base...") {
    const container = document.getElementById("chat-messages");
    const id = "loading-" + Date.now();
    const msgDiv = document.createElement("div");
    msgDiv.id = id;
    msgDiv.className = "message assistant-msg";
    msgDiv.innerHTML = `
        <div class="avatar">🤖</div>
        <div class="msg-body" style="color: var(--text-dim);">
            <div class="loading-dots">${escapeHtml(text)}</div>
        </div>
    `;
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeLoadingMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// --- Hybrid Search & Explorer ---
async function executeSearch() {
    const input = document.getElementById("search-query-input");
    const query = input.value.trim();
    if (!query) return;

    const topK = parseInt(document.getElementById("search-top-k").value) || 6;
    const container = document.getElementById("search-results-container");
    container.innerHTML = `<div class="empty-state"><p>Retrieving and ranking chunks...</p></div>`;

    try {
        const data = await safeFetchJson(`${API_BASE}/api/search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query, top_k: topK })
        });

        if (!data.hits || data.hits.length === 0) {
            container.innerHTML = `<div class="empty-state"><p>No relevant chunks found for '${escapeHtml(query)}'.</p></div>`;
            document.getElementById("prompt-preview-box").style.display = "none";
            return;
        }

        container.innerHTML = "";
        data.hits.forEach((hit, idx) => {
            const card = document.createElement("div");
            card.className = "chunk-card";
            card.innerHTML = `
                <div class="chunk-header">
                    <div>
                        <div class="chunk-title">${escapeHtml(hit.doc_title)}</div>
                        <div class="chunk-section">📍 ${escapeHtml(hit.section_path)}</div>
                    </div>
                    <div class="chunk-badges">
                        <span class="badge-score" title="Reciprocal Rank Fusion Score">RRF: ${hit.score.toFixed(5)}</span>
                        <span class="badge-score" title="BM25 Rank">BM25: #${hit.bm25_rank || '-'}</span>
                        <span class="badge-score" title="Vector Cosine Rank">Vec: #${hit.vector_rank || '-'}</span>
                    </div>
                </div>
                <div class="chunk-text">${escapeHtml(hit.content)}</div>
                <div style="font-size: 0.78rem; color: var(--text-dim);">
                    <a href="${escapeHtml(hit.doc_url)}" target="_blank" style="color: var(--accent); text-decoration: none;">🔗 Open Source URL</a>
                </div>
            `;
            container.appendChild(card);
        });

        // Show formatted prompt
        currentPrompt = data.prompt;
        document.getElementById("prompt-preview-text").innerText = data.prompt;
        document.getElementById("prompt-preview-box").style.display = "block";

    } catch (err) {
        container.innerHTML = `<div class="empty-state" style="color: var(--danger);"><p>Search error: ${err.message}</p></div>`;
    }
}

function copyFormattedPrompt() {
    if (!currentPrompt) return;
    navigator.clipboard.writeText(currentPrompt).then(() => {
        alert("Formatted grounded LLM prompt copied to clipboard!");
    });
}

// --- Knowledge Sources Hub ---
async function loadSources() {
    const grid = document.getElementById("sources-grid");
    grid.innerHTML = "<div class='empty-state'><p>Loading sources...</p></div>";

    try {
        const docs = await safeFetchJson(`${API_BASE}/api/documents`);

        loadStatus();

        if (!docs || docs.length === 0) {
            grid.innerHTML = "<div class='empty-state'><p>No documents indexed yet. Use the Ingest tab to add documentation.</p></div>";
            return;
        }

        grid.innerHTML = "";
        docs.forEach(doc => {
            const isPdf = (doc.url && doc.url.toLowerCase().endsWith(".pdf")) ||
                          (doc.url && doc.url.toLowerCase().includes(".pdf")) ||
                          (doc.description && doc.description.includes("PDF"));

            const typeBadge = isPdf 
                ? `<span class="badge-score" style="background: rgba(239, 68, 68, 0.15); color: #f87171; border-color: rgba(239, 68, 68, 0.3);">📄 PDF</span>`
                : `<span class="badge-score" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border-color: rgba(56, 189, 248, 0.3);">🌐 Web</span>`;

            const card = document.createElement("div");
            card.className = "source-card";
            card.innerHTML = `
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 6px;">
                        <h4 style="margin: 0;">${escapeHtml(doc.headline || doc.title)}</h4>
                        ${typeBadge}
                    </div>
                    <p class="source-desc">${escapeHtml(doc.description || "Technical documentation")}</p>
                </div>
                <div>
                    <div class="source-meta">
                        <span>📦 ${doc.chunk_count} Chunks</span>
                        <span>📝 ${doc.word_count} Words</span>
                    </div>
                    <div class="source-actions">
                        <button class="btn btn-xs btn-outline" onclick="viewDocDetail('${encodeURIComponent(doc.url)}')">View Content</button>
                        <button class="btn btn-xs btn-danger" onclick="deleteDoc('${encodeURIComponent(doc.url)}')">Delete</button>
                    </div>
                </div>
            `;
            grid.appendChild(card);
        });
    } catch (err) {
        grid.innerHTML = `<div class='empty-state' style='color: var(--danger);'><p>Error loading sources: ${err.message}</p></div>`;
    }
}

async function viewDocDetail(encodedUrl) {
    const url = decodeURIComponent(encodedUrl);
    try {
        const doc = await safeFetchJson(`${API_BASE}/api/documents/detail?url=${encodeURIComponent(url)}`);

        document.getElementById("modal-title").innerText = doc.headline || doc.title;
        document.getElementById("modal-meta").innerHTML = `
            <p style="font-size: 0.82rem; color: var(--text-dim); margin-bottom: 8px;">
                <strong>URL:</strong> <a href="${doc.url}" target="_blank" style="color: var(--accent);">${doc.url}</a> | 
                <strong>Chunks:</strong> ${doc.chunk_count} | 
                <strong>Words:</strong> ${doc.word_count}
            </p>
        `;
        document.getElementById("modal-body").innerHTML = (typeof marked !== "undefined")
            ? marked.parse(doc.markdown_content)
            : `<pre>${escapeHtml(doc.markdown_content)}</pre>`;

        document.getElementById("doc-modal").style.display = "flex";
    } catch (err) {
        alert("Failed to load document: " + err.message);
    }
}

function closeModal() {
    document.getElementById("doc-modal").style.display = "none";
}

async function deleteDoc(encodedUrl) {
    const url = decodeURIComponent(encodedUrl);
    if (!confirm(`Are you sure you want to delete this document from the knowledge base?\n${url}`)) return;

    try {
        await safeFetchJson(`${API_BASE}/api/documents?url=${encodeURIComponent(url)}`, {
            method: "DELETE"
        });
        loadSources();
    } catch (err) {
        alert("Delete error: " + err.message);
    }
}

// --- Ingest URLs & Presets ---
function setBitbucketPreset() {
    document.getElementById("ingest-urls-textarea").value = [
        "https://support.atlassian.com/bitbucket-cloud/docs/what-is-a-workspace/",
        "https://support.atlassian.com/bitbucket-cloud/docs/create-your-workspace/",
        "https://support.atlassian.com/bitbucket-cloud/docs/grant-access-to-a-workspace/"
    ].join("\n");
}

function setEsgPreset() {
    document.getElementById("ingest-urls-textarea").value = [
        "https://www.pwc.com/sk/en/environmental-social-and-corporate-governance-esg/esg-reporting.html",
        "https://www.sustainabilityreports.com/services/esg-data"
    ].join("\n");
}

async function submitIngest() {
    const textarea = document.getElementById("ingest-urls-textarea");
    const rawUrls = textarea.value.split("\n").map(u => u.trim()).filter(u => u.length > 0);

    if (rawUrls.length === 0) {
        alert("Please enter at least one valid URL to ingest.");
        return;
    }

    const logBox = document.getElementById("ingest-status-log");
    const logContent = document.getElementById("ingest-log-content");
    const btn = document.getElementById("ingest-submit-btn");

    logBox.style.display = "block";
    logContent.innerHTML = `<div>⏳ Connecting to scraper and parsing ${rawUrls.length} URL(s)...</div>`;
    btn.disabled = true;

    try {
        const data = await safeFetchJson(`${API_BASE}/api/ingest`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ urls: rawUrls })
        });

        let logs = "";
        data.results.forEach(res => {
            if (res.status === "success") {
                logs += `<div>✅ [SUCCESS] Ingested '${escapeHtml(res.headline)}' (${res.chunks} chunks, ${res.word_count} words)</div>`;
            } else {
                logs += `<div style="color: var(--danger);">❌ [FAILED] ${escapeHtml(res.url)}: ${escapeHtml(res.error)}</div>`;
            }
        });
        logContent.innerHTML = logs;
        loadStatus();
    } catch (err) {
        logContent.innerHTML += `<div style="color: var(--danger);">❌ Ingestion request failed: ${err.message}</div>`;
    } finally {
        btn.disabled = false;
    }
}

// --- PDF & File Ingestion ---
function handleFileSelect(files) {
    if (!files || files.length === 0) return;
    const file = files[0];
    uploadPdfDocument(file);
}

async function uploadPdfDocument(file) {
    if (!file) return;

    const progressBox = document.getElementById("file-upload-progress");
    const nameEl = document.getElementById("upload-filename");
    const badgeEl = document.getElementById("upload-status-badge");
    const fillEl = document.getElementById("upload-progress-bar");
    const msgEl = document.getElementById("upload-result-msg");

    if (progressBox) progressBox.style.display = "block";
    if (nameEl) nameEl.innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    if (badgeEl) {
        badgeEl.className = "badge-status";
        badgeEl.innerText = "Extracting & Indexing...";
    }
    if (fillEl) fillEl.style.width = "45%";
    if (msgEl) {
        msgEl.className = "upload-result-msg";
        msgEl.innerText = "Extracting pages, parsing tables, and generating semantic chunks...";
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const resp = await fetch(`${API_BASE}/api/ingest/file`, {
            method: "POST",
            body: formData,
        });

        const data = await resp.json();

        if (!resp.ok) {
            throw new Error(data.detail || `HTTP ${resp.status}`);
        }

        if (fillEl) fillEl.style.width = "100%";
        if (badgeEl) {
            badgeEl.className = "badge-status success";
            badgeEl.innerText = "Indexed";
        }
        if (msgEl) {
            msgEl.className = "upload-result-msg success";
            msgEl.innerHTML = `✅ Successfully ingested <strong>${escapeHtml(data.headline)}</strong>: ${data.chunks} chunks created (${data.word_count} words).`;
        }

        loadStatus();
        loadSources();
    } catch (err) {
        if (fillEl) fillEl.style.width = "100%";
        if (badgeEl) {
            badgeEl.className = "badge-status error";
            badgeEl.innerText = "Failed";
        }
        if (msgEl) {
            msgEl.className = "upload-result-msg error";
            msgEl.innerText = `❌ Error indexing document: ${err.message}`;
        }
    }
}

function setupDropZone() {
    const dropZone = document.getElementById("pdf-drop-zone");
    if (!dropZone) return;

    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove("dragover");
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            uploadPdfDocument(files[0]);
        }
    }, false);
}

// --- Activity & MCP Logs ---
async function loadLogs() {
    const terminal = document.getElementById("logs-terminal");
    const catSelect = document.getElementById("log-category-select");
    const category = catSelect ? catSelect.value : "";

    const url = category ? `${API_BASE}/api/logs?category=${encodeURIComponent(category)}` : `${API_BASE}/api/logs`;

    try {
        const data = await safeFetchJson(url);
        const logs = data.logs || [];

        if (logs.length === 0) {
            terminal.innerHTML = "<div class='empty-state'><p>No activity logs recorded yet. MCP calls from Gemini and searches will appear here in real time.</p></div>";
            return;
        }

        terminal.innerHTML = "";
        logs.forEach((entry, idx) => {
            const row = document.createElement("div");
            row.className = "log-entry";

            const timeStr = entry.timestamp ? entry.timestamp.substring(11, 19) : "";
            const cat = entry.category || "SYS";
            const latStr = entry.latency_ms !== null && entry.latency_ms !== undefined ? `${entry.latency_ms}ms` : "";

            let detailsJson = "";
            if (entry.details && Object.keys(entry.details).length > 0) {
                detailsJson = JSON.stringify(entry.details, null, 2);
            }

            row.innerHTML = `
                <div class="log-row-main">
                    <span class="log-time">${escapeHtml(timeStr)}</span>
                    <span class="badge-cat badge-cat-${escapeHtml(cat)}">${escapeHtml(cat)}</span>
                    <span class="log-action">${escapeHtml(entry.action || entry.event_type)}</span>
                    ${latStr ? `<span class="log-latency">${escapeHtml(latStr)}</span>` : ""}
                </div>
                ${detailsJson ? `<pre class="log-details-block">${escapeHtml(detailsJson)}</pre>` : ""}
            `;
            terminal.appendChild(row);
        });

    } catch (err) {
        terminal.innerHTML = `<div class='empty-state' style='color: var(--danger);'><p>Error loading logs: ${err.message}</p></div>`;
    }
}

async function clearAllLogs() {
    if (!confirm("Are you sure you want to clear all activity logs?")) return;
    try {
        await safeFetchJson(`${API_BASE}/api/logs`, { method: "DELETE" });
        loadLogs();
    } catch (err) {
        alert("Failed to clear logs: " + err.message);
    }
}

function escapeHtml(text) {
    if (!text) return "";
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}
