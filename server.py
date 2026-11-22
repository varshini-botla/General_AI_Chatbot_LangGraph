"""
Web Server and Interactive Chat Interface for General-Purpose AI Chatbot
and Autonomous Market Research Agent (Agentic RAG).
Serves a responsive ChatGPT-style UI at '/' and connects directly to LangGraph StateGraphs.
"""

import os
import uuid
from typing import Optional, List
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from agent import create_graph
from market_research.agent import create_market_research_graph

load_dotenv()

app = FastAPI(title="General-Purpose AI Chatbot & Market Research Agent", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent SQLite Checkpointers for web sessions
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 1. General Chatbot Graph
conn_general = sqlite3.connect(os.path.join(DATA_DIR, "web_sessions.sqlite"), check_same_thread=False)
checkpointer_general = SqliteSaver(conn_general)
bot_graph = create_graph(checkpointer=checkpointer_general)

# 2. Market Research Agentic RAG Graph
conn_mr = sqlite3.connect(os.path.join(DATA_DIR, "web_mr_sessions.sqlite"), check_same_thread=False)
checkpointer_mr = SqliteSaver(conn_mr)
market_graph = create_market_research_graph(checkpointer=checkpointer_mr)


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    mode: Optional[str] = "general"  # 'general' or 'market_research'


class MarketResearchRequest(BaseModel):
    topic: str
    thread_id: Optional[str] = None


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>General AI & Market Research Assistant</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: #0f172a;
      color: #f1f5f9;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: #1e293b;
      padding: 14px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #334155;
      flex-wrap: wrap;
      gap: 12px;
    }
    .header-left {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .header-title {
      font-weight: 700;
      font-size: 1.15rem;
      letter-spacing: -0.02em;
    }
    .badge {
      font-size: 0.72rem;
      padding: 3px 8px;
      background: #3b82f6;
      border-radius: 9999px;
      text-transform: uppercase;
      font-weight: 600;
    }
    .badge-rag {
      background: #8b5cf6;
      color: #fff;
    }
    .mode-switcher {
      display: flex;
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 10px;
      padding: 3px;
      gap: 4px;
    }
    .mode-btn {
      background: transparent;
      border: none;
      color: #94a3b8;
      padding: 6px 14px;
      border-radius: 7px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }
    .mode-btn.active {
      background: #2563eb;
      color: #ffffff;
      box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    .mode-btn#mode-mr.active {
      background: #7c3aed;
    }
    .new-btn {
      background: #334155;
      color: #e2e8f0;
      border: 1px solid #475569;
      padding: 7px 13px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.82rem;
      font-weight: 500;
      transition: all 0.15s;
    }
    .new-btn:hover { background: #475569; color: white; }
    #chat-container {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-width: 880px;
      width: 100%;
      margin: 0 auto;
    }
    .msg {
      display: flex;
      gap: 12px;
      max-width: 88%;
      line-height: 1.6;
      font-size: 0.95rem;
      word-break: break-word;
    }
    .msg.user {
      align-self: flex-end;
      flex-direction: row-reverse;
    }
    .msg.assistant {
      align-self: flex-start;
    }
    .avatar {
      width: 34px;
      height: 34px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.85rem;
      font-weight: bold;
      flex-shrink: 0;
    }
    .msg.user .avatar { background: #3b82f6; color: white; }
    .msg.assistant .avatar { background: #10b981; color: white; }
    .msg.assistant.mr-avatar .avatar { background: #8b5cf6; color: white; }
    .bubble {
      padding: 14px 18px;
      border-radius: 12px;
      white-space: pre-wrap;
    }
    .msg.user .bubble {
      background: #2563eb;
      color: white;
      border-bottom-right-radius: 2px;
    }
    .msg.assistant .bubble {
      background: #1e293b;
      color: #f8fafc;
      border: 1px solid #334155;
      border-bottom-left-radius: 2px;
    }
    .citations-box {
      margin-top: 10px;
      padding: 10px 12px;
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      font-size: 0.8rem;
      color: #94a3b8;
    }
    .citations-box strong { color: #cbd5e1; }
    .grounding-tag {
      display: inline-block;
      margin-top: 8px;
      font-size: 0.72rem;
      padding: 2px 8px;
      background: #065f46;
      color: #6ee7b7;
      border-radius: 6px;
      font-weight: 600;
    }
    .welcome-card {
      text-align: center;
      margin: auto;
      padding: 32px 20px;
      max-width: 580px;
    }
    .welcome-card h2 { font-size: 1.6rem; margin-bottom: 8px; color: #f8fafc; }
    .welcome-card p { color: #94a3b8; font-size: 0.95rem; margin-bottom: 24px; line-height: 1.5; }
    .suggestions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .chip {
      background: #1e293b;
      border: 1px solid #334155;
      padding: 12px;
      border-radius: 10px;
      font-size: 0.85rem;
      text-align: left;
      cursor: pointer;
      color: #cbd5e1;
      transition: all 0.15s;
    }
    .chip:hover { background: #334155; color: #ffffff; border-color: #64748b; }
    #input-bar {
      background: #1e293b;
      padding: 16px 24px;
      border-top: 1px solid #334155;
    }
    .input-form {
      max-width: 880px;
      margin: 0 auto;
      display: flex;
      gap: 10px;
      position: relative;
    }
    textarea {
      flex: 1;
      background: #0f172a;
      border: 1px solid #475569;
      color: #f8fafc;
      padding: 12px 16px;
      border-radius: 10px;
      resize: none;
      height: 48px;
      font-size: 0.95rem;
      font-family: inherit;
      outline: none;
      transition: border 0.15s;
    }
    textarea:focus { border-color: #3b82f6; }
    button.send-btn {
      background: #3b82f6;
      color: white;
      border: none;
      padding: 0 20px;
      border-radius: 10px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.15s;
    }
    button.send-btn:hover { background: #2563eb; }
    button.send-btn:disabled { background: #475569; cursor: not-allowed; }
    .footer-note {
      text-align: center;
      font-size: 0.72rem;
      color: #64748b;
      margin-top: 8px;
    }
  </style>
</head>
<body>
  <header>
    <div class="header-left">
      <span class="header-title" id="page-title">General AI Assistant</span>
      <span class="badge">Groq 120B</span>
      <span class="badge" style="background:#10b981;">LangGraph</span>
      <span class="badge badge-rag" id="rag-badge" style="display:none;">Agentic RAG Active</span>
    </div>
    
    <div class="mode-switcher">
      <button class="mode-btn active" id="mode-general" onclick="setMode('general')">💬 General Chat</button>
      <button class="mode-btn" id="mode-mr" onclick="setMode('market_research')">📊 Market Research Agent</button>
    </div>

    <button class="new-btn" onclick="startNewThread()">+ New Session</button>
  </header>

  <div id="chat-container">
    <div class="welcome-card" id="welcome">
      <h2 id="welcome-title">How can I help you today?</h2>
      <p id="welcome-desc">Powered by LangGraph StateGraph & Groq high-speed inference.</p>
      <div class="suggestions" id="suggestions-box">
        <!-- Rendered dynamically -->
      </div>
    </div>
  </div>

  <div id="input-bar">
    <form class="input-form" id="chat-form" onsubmit="sendMessage(event)">
      <textarea id="prompt" placeholder="Ask anything (e.g. science, Python, mathematics, history)..." onkeydown="handleKey(event)"></textarea>
      <button type="submit" class="send-btn" id="send-btn">Send</button>
    </form>
    <div class="footer-note" id="footer-note">LangGraph Agent running with Groq inference & LangSmith Tracing</div>
  </div>

  <script>
    let currentMode = 'general';
    let threadId = localStorage.getItem('ai_chat_thread_id') || ('thread_' + Math.random().toString(36).substring(2, 11));
    localStorage.setItem('ai_chat_thread_id', threadId);

    const chatContainer = document.getElementById('chat-container');
    const promptInput = document.getElementById('prompt');
    const sendBtn = document.getElementById('send-btn');
    const welcome = document.getElementById('welcome');
    const welcomeTitle = document.getElementById('welcome-title');
    const welcomeDesc = document.getElementById('welcome-desc');
    const suggestionsBox = document.getElementById('suggestions-box');
    const ragBadge = document.getElementById('rag-badge');
    const pageTitle = document.getElementById('page-title');
    const footerNote = document.getElementById('footer-note');

    const GENERAL_SUGGESTIONS = [
      { text: "Explain Artificial Intelligence in simple terms.", label: "💡 What is AI?" },
      { text: "Write a Python function to check for prime numbers.", label: "🐍 Python Prime Check" },
      { text: "What is quantum computing and how do qubits work?", label: "⚛️ Quantum Computing" },
      { text: "Give me 3 practical tips for coding interviews.", label: "🎯 Interview Prep Tips" }
    ];

    const MR_SUGGESTIONS = [
      { text: "Autonomous AI Agents and Workflow Automation in Enterprise Software", label: "📈 Enterprise AI Agents Market" },
      { text: "Cloud AI Semiconductor & Specialized Inference Chips (Groq, NVIDIA, TPU)", label: "⚡ Cloud AI Chips & GPUs" },
      { text: "Global Electric Vehicle and Autonomous Driving Technology Landscape", label: "🚗 EV & Autonomous Driving" },
      { text: "Cybersecurity & AI-Driven Threat Detection Market Forecast", label: "🔒 Cybersecurity AI Threat Defense" }
    ];

    function renderSuggestions() {
      suggestionsBox.innerHTML = '';
      const list = currentMode === 'general' ? GENERAL_SUGGESTIONS : MR_SUGGESTIONS;
      list.forEach(item => {
        const div = document.createElement('div');
        div.className = 'chip';
        div.innerText = item.label;
        div.onclick = () => quickSend(item.text);
        suggestionsBox.appendChild(div);
      });
    }

    function setMode(mode) {
      currentMode = mode;
      document.getElementById('mode-general').className = 'mode-btn' + (mode === 'general' ? ' active' : '');
      document.getElementById('mode-mr').className = 'mode-btn' + (mode === 'market_research' ? ' active' : '');

      if (mode === 'market_research') {
        pageTitle.innerText = "Market Research Agent";
        ragBadge.style.display = "inline-block";
        welcomeTitle.innerText = "Autonomous Market Research Agent";
        welcomeDesc.innerText = "Implementing Agentic RAG Techniques: Dynamic Query Planning ➔ Multi-Source Hybrid Retrieval ➔ Self-Reflective Document Grading ➔ Executive Report Synthesis.";
        promptInput.placeholder = "Enter a market research topic (e.g. AI Agents, Cloud GPUs, CleanTech, Autonomous Driving)...";
        footerNote.innerText = "Agentic RAG Pipeline: LangGraph + Hybrid BM25/TF-IDF Retrieval + Groq 120B Inference";
      } else {
        pageTitle.innerText = "General AI Assistant";
        ragBadge.style.display = "none";
        welcomeTitle.innerText = "How can I help you today?";
        welcomeDesc.innerText = "Powered by LangGraph StateGraph & Groq high-speed inference.";
        promptInput.placeholder = "Ask anything (e.g. science, Python, mathematics, history)...";
        footerNote.innerText = "LangGraph Agent running with Groq inference & LangSmith Tracing";
      }
      renderSuggestions();
    }

    // Initial render
    renderSuggestions();

    function handleKey(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage(e);
      }
    }

    function quickSend(text) {
      promptInput.value = text;
      sendMessage(new Event('submit'));
    }

    function startNewThread() {
      threadId = 'thread_' + Math.random().toString(36).substring(2, 11);
      localStorage.setItem('ai_chat_thread_id', threadId);
      chatContainer.innerHTML = '';
      chatContainer.appendChild(welcome);
    }

    function appendMessage(role, text, metadata = null) {
      if (welcome && welcome.parentElement) {
        welcome.remove();
      }
      const msgDiv = document.createElement('div');
      msgDiv.className = `msg ${role}` + (role === 'assistant' && currentMode === 'market_research' ? ' mr-avatar' : '');
      
      const avatar = document.createElement('div');
      avatar.className = 'avatar';
      avatar.innerText = role === 'user' ? 'U' : (currentMode === 'market_research' ? 'MR' : 'AI');

      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.innerText = text;

      if (metadata && metadata.citations && metadata.citations.length > 0) {
        const citDiv = document.createElement('div');
        citDiv.className = 'citations-box';
        citDiv.innerHTML = `<strong>📚 Sources Consulted:</strong><br>` + metadata.citations.map(c => `• ${c}`).join('<br>');
        bubble.appendChild(citDiv);
      }

      if (metadata && metadata.grounding_status) {
        const tag = document.createElement('div');
        tag.className = 'grounding-tag';
        tag.innerText = `✓ ${metadata.grounding_status}`;
        bubble.appendChild(tag);
      }

      msgDiv.appendChild(avatar);
      msgDiv.appendChild(bubble);
      chatContainer.appendChild(msgDiv);
      chatContainer.scrollTop = chatContainer.scrollHeight;
      return bubble;
    }

    async function sendMessage(e) {
      if (e) e.preventDefault();
      const text = promptInput.value.trim();
      if (!text) return;

      appendMessage('user', text);
      promptInput.value = '';
      sendBtn.disabled = true;

      const thinkingMsg = currentMode === 'market_research' 
        ? 'Conducting Agentic RAG Market Research (Decomposing queries ➔ Hybrid retrieval ➔ Grading evidence ➔ Synthesizing report)...'
        : 'Thinking...';
      const thinkingBubble = appendMessage('assistant', thinkingMsg);

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, thread_id: threadId, mode: currentMode })
        });
        const data = await response.json();
        
        thinkingBubble.innerText = data.response;

        if (data.citations && data.citations.length > 0) {
          const citDiv = document.createElement('div');
          citDiv.className = 'citations-box';
          citDiv.innerHTML = `<strong>📚 Sources Consulted:</strong><br>` + data.citations.map(c => `• ${c}`).join('<br>');
          thinkingBubble.appendChild(citDiv);
        }

        if (data.grounding_status) {
          const tag = document.createElement('div');
          tag.className = 'grounding-tag';
          tag.innerText = `✓ ${data.grounding_status}`;
          thinkingBubble.appendChild(tag);
        }

      } catch (err) {
        thinkingBubble.innerText = 'Error: Failed to reach the chatbot server. Please try again.';
      } finally {
        sendBtn.disabled = false;
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }
    }
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serves the interactive ChatGPT-style and Market Research web interface."""
    return HTML_PAGE

@app.get("/health")
@app.get("/info")
async def health_check():
    """Health check endpoint exposing system status and available agents."""
    return {
        "ok": True,
        "status": "online",
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        "agents": [
            "general_chatbot",
            "market_research_agentic_rag",
        ],
        "agentic_rag_techniques": [
            "dynamic_query_planning",
            "hybrid_bm25_tfidf_retrieval",
            "self_reflective_document_grading",
            "iterative_query_reformulation",
            "executive_market_synthesis",
            "grounding_verification",
        ],
    }

@app.post("/api/market-research")
async def market_research_endpoint(req: MarketResearchRequest):
    """
    Dedicated endpoint to execute the Agentic RAG Market Research workflow on a specific topic.
    Returns structured executive report, citations, and grounding status.
    """
    t_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": t_id}}

    initial_state = {
        "messages": [HumanMessage(content=req.topic)],
        "research_topic": req.topic,
        "sub_queries": [],
        "retrieved_docs": [],
        "graded_docs": [],
        "iteration": 0,
        "max_iterations": 2,
        "synthesis_report": "",
        "grounding_status": "",
        "citations": [],
    }

    result = market_graph.invoke(initial_state, config=config)
    ai_answer = result["messages"][-1].content
    return {
        "topic": req.topic,
        "report": ai_answer,
        "citations": result.get("citations", []),
        "grounding_status": result.get("grounding_status", ""),
        "thread_id": t_id,
    }

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Unified chat endpoint.
    Routes to general chatbot or Agentic RAG Market Research Agent based on req.mode.
    """
    t_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": t_id}}

    if req.mode == "market_research":
        initial_state = {
            "messages": [HumanMessage(content=req.message)],
            "research_topic": req.message,
            "sub_queries": [],
            "retrieved_docs": [],
            "graded_docs": [],
            "iteration": 0,
            "max_iterations": 2,
            "synthesis_report": "",
            "grounding_status": "",
            "citations": [],
        }
        result = market_graph.invoke(initial_state, config=config)
        ai_answer = result["messages"][-1].content
        return {
            "response": ai_answer,
            "thread_id": t_id,
            "mode": "market_research",
            "citations": result.get("citations", []),
            "grounding_status": result.get("grounding_status", ""),
        }
    else:
        result = bot_graph.invoke(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
        )
        ai_answer = result["messages"][-1].content
        return {
            "response": ai_answer,
            "thread_id": t_id,
            "mode": "general",
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
