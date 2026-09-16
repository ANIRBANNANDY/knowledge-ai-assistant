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

// --- Status Loading ---
async function loadStatus() {
    try {
        const data = await safeFetchJson(`${API_BASE}/api/status`);
        document.getElementById("stat-docs").innerText = data.total_documents || 0;
        document.getElementById("stat-chunks").innerText = data.total_chunks || 0;
        document.getElementById("stat-words").innerText = (data.total_words || 0).toLocaleString();
    } catch (err) {
        console.error("Failed to load status:", err);
    }
}

// --- Chat Assistant ---
async function handleChatSubmit(e) {
    e.preventDefault();
    const input = document.getElementById("chat-input");
    const query = input.value.trim();
    if (!query) return;

    appendMessage("user", query);
    input.value = "";

    const loadingId = appendLoadingMessage();

    try {
        const data = await safeFetchJson(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: query, top_k: 6 })
        });
        removeLoadingMessage(loadingId);
        appendAssistantMessage(data.reply, data.citations, data.llm_prompt);
        loadStatus();
    } catch (err) {
        removeLoadingMessage(loadingId);
        appendMessage("assistant", "⚠️ Error communicating with local knowledge assistant: " + err.message);
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

function appendAssistantMessage(replyMarkdown, citations, promptText) {
    const container = document.getElementById("chat-messages");
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant-msg";

    const parsedHtml = (typeof marked !== "undefined") ? marked.parse(replyMarkdown) : replyMarkdown;

    let citationsHtml = "";
    if (citations && citations.length > 0) {
        citationsHtml = `
            <div class="citations-box">
                <div class="citations-header">Sources & Citations:</div>
                ${citations.map(c => `
                    <a href="${escapeHtml(c.url)}" target="_blank" rel="noopener noreferrer" class="citation-link">
                        <span>🔗 ${escapeHtml(c.title)} (${escapeHtml(c.section)})</span>
                    </a>
                `).join("")}
            </div>
        `;
    }

    msgDiv.innerHTML = `
        <div class="avatar">🤖</div>
        <div class="msg-body">
            <div class="markdown-content">${parsedHtml}</div>
            ${citationsHtml}
        </div>
    `;
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function appendLoadingMessage() {
    const container = document.getElementById("chat-messages");
    const id = "loading-" + Date.now();
    const msgDiv = document.createElement("div");
    msgDiv.id = id;
    msgDiv.className = "message assistant-msg";
    msgDiv.innerHTML = `
        <div class="avatar">🤖</div>
        <div class="msg-body" style="color: var(--text-dim);">
            <div class="loading-dots">Searching local knowledge base...</div>
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
            const card = document.createElement("div");
            card.className = "source-card";
            card.innerHTML = `
                <div>
                    <h4>${escapeHtml(doc.headline || doc.title)}</h4>
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

// --- Ingest URLs ---
function setBitbucketPreset() {
    document.getElementById("ingest-urls-textarea").value = [
        "https://support.atlassian.com/bitbucket-cloud/docs/what-is-a-workspace/",
        "https://support.atlassian.com/bitbucket-cloud/docs/create-your-workspace/",
        "https://support.atlassian.com/bitbucket-cloud/docs/grant-access-to-a-workspace/"
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
