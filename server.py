"""
Web Server and Interactive Chat Interface for General-Purpose AI Chatbot.
Serves a responsive ChatGPT-style UI at '/' and connects directly to LangGraph agent.py StateGraph.
"""

import os
import uuid
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from agent import create_graph

load_dotenv()

app = FastAPI(title="General-Purpose AI Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent SQLite Checkpointer for web sessions
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "web_sessions.sqlite")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)
bot_graph = create_graph(checkpointer=checkpointer)

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>General AI Chatbot</title>
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
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #334155;
    }
    .header-title {
      display: flex;
      align-items: center;
      gap: 12px;
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
    .new-btn {
      background: #334155;
      color: #e2e8f0;
      border: 1px solid #475569;
      padding: 8px 14px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.85rem;
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
      max-width: 850px;
      width: 100%;
      margin: 0 auto;
    }
    .msg {
      display: flex;
      gap: 12px;
      max-width: 85%;
      line-height: 1.55;
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
    .bubble {
      padding: 12px 16px;
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
    .welcome-card {
      text-align: center;
      margin: auto;
      padding: 32px 20px;
      max-width: 520px;
    }
    .welcome-card h2 { font-size: 1.6rem; margin-bottom: 8px; color: #f8fafc; }
    .welcome-card p { color: #94a3b8; font-size: 0.95rem; margin-bottom: 24px; }
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
      max-width: 850px;
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
    <div class="header-title">
      <span>General AI Assistant</span>
      <span class="badge">Groq 120B</span>
      <span class="badge" style="background:#10b981;">LangGraph</span>
    </div>
    <button class="new-btn" onclick="startNewThread()">+ New Chat</button>
  </header>

  <div id="chat-container">
    <div class="welcome-card" id="welcome">
      <h2>How can I help you today?</h2>
      <p>Powered by LangGraph StateGraph & Groq high-speed inference.</p>
      <div class="suggestions">
        <div class="chip" onclick="quickSend('Explain Artificial Intelligence in simple terms.')">💡 What is AI?</div>
        <div class="chip" onclick="quickSend('Write a Python function to check for prime numbers.')">🐍 Python Prime Check</div>
        <div class="chip" onclick="quickSend('What is quantum computing and how do qubits work?')">⚛️ Quantum Computing</div>
        <div class="chip" onclick="quickSend('Give me 3 practical tips for coding interviews.')">🎯 Interview Prep Tips</div>
      </div>
    </div>
  </div>

  <div id="input-bar">
    <form class="input-form" id="chat-form" onsubmit="sendMessage(event)">
      <textarea id="prompt" placeholder="Ask anything (e.g. science, Python, mathematics, history)..." onkeydown="handleKey(event)"></textarea>
      <button type="submit" class="send-btn" id="send-btn">Send</button>
    </form>
    <div class="footer-note">LangGraph Agent running live on Render Cloud with LangSmith Tracing</div>
  </div>

  <script>
    let threadId = localStorage.getItem('ai_chat_thread_id') || ('thread_' + Math.random().toString(36).substring(2, 11));
    localStorage.setItem('ai_chat_thread_id', threadId);

    const chatContainer = document.getElementById('chat-container');
    const promptInput = document.getElementById('prompt');
    const sendBtn = document.getElementById('send-btn');
    const welcome = document.getElementById('welcome');

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

    function appendMessage(role, text) {
      if (welcome && welcome.parentElement) {
        welcome.remove();
      }
      const msgDiv = document.createElement('div');
      msgDiv.className = `msg ${role}`;
      
      const avatar = document.createElement('div');
      avatar.className = 'avatar';
      avatar.innerText = role === 'user' ? 'U' : 'AI';

      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.innerText = text;

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

      const thinkingBubble = appendMessage('assistant', 'Thinking...');

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, thread_id: threadId })
        });
        const data = await response.json();
        thinkingBubble.innerText = data.response;
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
    """Serves the ChatGPT-style web interface."""
    return HTML_PAGE

@app.get("/health")
@app.get("/info")
async def health_check():
    """Health check endpoint."""
    return {"ok": True, "status": "online", "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")}

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Invokes the LangGraph StateGraph with thread-level memory."""
    t_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": t_id}}
    
    result = bot_graph.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )
    ai_answer = result["messages"][-1].content
    return {"response": ai_answer, "thread_id": t_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
