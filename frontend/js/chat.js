/**
 * Chat client — talks to the RAG API (separate process).
 * Override base URL: window.RAG_API_BASE = "https://api.example.com/api"
 */
(function () {
  const API_BASE = (window.RAG_API_BASE || "http://localhost:8000/api").replace(/\/$/, "");

  const thread = document.getElementById("thread");
  const empty = document.getElementById("empty");
  const form = document.getElementById("form");
  const questionEl = document.getElementById("question");
  const strategyEl = document.getElementById("strategy");
  const sendBtn = document.getElementById("send");
  const dot = document.getElementById("dot");
  const statusText = document.getElementById("statusText");

  async function refreshHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      if (data.index_ready) {
        dot.className = "dot ok";
        statusText.textContent = "Index ready · " + data.collection;
      } else {
        dot.className = "dot bad";
        statusText.textContent = "Index empty — run ./scripts/ingest_documents.sh";
      }
    } catch {
      dot.className = "dot bad";
      statusText.textContent = "API unreachable (" + API_BASE + ")";
    }
  }

  function addMessage(role, text, meta) {
    if (empty) empty.remove();
    const wrap = document.createElement("div");
    wrap.className = "msg " + role;
    if (role === "user") {
      wrap.textContent = text;
    } else {
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = text;
      wrap.appendChild(bubble);
      if (meta) {
        const m = document.createElement("div");
        m.className = "meta";
        m.textContent = meta;
        wrap.appendChild(m);
      }
    }
    thread.appendChild(wrap);
    thread.scrollTop = thread.scrollHeight;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = questionEl.value.trim();
    if (!question) return;

    const strategyValue = strategyEl.value;
    const useRouting = strategyValue === "routing";
    const body = {
      question,
      use_routing: useRouting,
      strategy: useRouting ? null : strategyValue,
    };

    addMessage("user", question);
    questionEl.value = "";
    sendBtn.disabled = true;
    sendBtn.textContent = "…";

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail;
        throw new Error(typeof detail === "string" ? detail : "Request failed");
      }
      const n = (data.sources || []).length;
      addMessage(
        "bot",
        data.answer,
        "strategy: " + data.strategy + (n ? " · " + n + " source" + (n === 1 ? "" : "s") : "")
      );
    } catch (err) {
      addMessage("bot", "Error: " + (err.message || err));
    } finally {
      sendBtn.disabled = false;
      sendBtn.textContent = "Send";
      questionEl.focus();
    }
  });

  questionEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  refreshHealth();
})();
