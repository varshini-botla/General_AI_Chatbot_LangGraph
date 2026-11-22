# General-Purpose AI Chatbot (LangGraph + Groq + LangSmith)

A production-ready, general-purpose conversational AI assistant built with **LangGraph**, powered by **Groq** high-speed LLM inference, and comprehensively monitored with **LangSmith**.

---

## 1. Project Overview
This chatbot behaves as a versatile, ChatGPT-style AI assistant capable of answering questions, writing and debugging code (Python, Java, JavaScript, etc.), explaining academic and scientific concepts, reasoning logically, and remembering conversation context across multiple turns.

Key characteristics:
- **Framework**: LangGraph `StateGraph` with stateful message reducers (`add_messages`).
- **Inference Engine**: Groq Cloud API using currently supported high-performance models (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`).
- **Observability**: Native LangSmith automatic tracing under the `General-AI-Chatbot` project.
- **Local Dev & Visualization**: LangGraph CLI (`langgraph dev`) and LangGraph Studio.
- **State Persistence**: SQLite-backed checkpointer locally (`data/local_checkpoints.sqlite`), managed/Postgres-ready for cloud.

---

## 2. Architecture

### Local Development Architecture
```text
User Input
    │
    ▼
LangGraph Studio (Browser UI)
    │
    ▼
LangGraph CLI Dev Server (127.0.0.1:2024)
    │
    ▼
LangGraph StateGraph (agent.py)
    │
    ▼
Chatbot Node (prepends system prompt & message history)
    │
    ▼
ChatGroq (openai/gpt-oss-120b)
    │
    ▼
Groq API Inference ─────────► LangSmith Automatic Tracing (General-AI-Chatbot)
    │
    ▼
AI Response appended to State
    │
    ▼
Local SQLite Checkpointer (data/local_checkpoints.sqlite)
```

### Cloud Production Architecture
```text
External Client / Web Frontend
    │
    ▼
Hosted LangGraph Agent Server (LangGraph Cloud / Container)
    │
    ▼
StateGraph Workflow (same agent.py graph)
    │
    ▼
Groq API ────────────────────► LangSmith Production Telemetry
    │
    ▼
Production Checkpoint Store (Managed PostgreSQL / Cloud Checkpointer)
```

---

## 3. Prerequisites
- **Operating System**: Windows 10/11 (PowerShell), macOS, or Linux
- **Python**: Version 3.11 or 3.12 (`python --version`)
- **Git**: Version 2.30+ (`git --version`)
- **Groq API Key**: Obtainable from [console.groq.com](https://console.groq.com)
- **LangSmith Account & API Key**: Obtainable from [smith.langchain.com](https://smith.langchain.com)

---

## 4. Local Setup Guide (Windows PowerShell)

### Step 1: Open Terminal and Navigate to Project
```powershell
Set-Location -Path "C:\projects\AI CHATBOT\General_AI_Chatbot_LangGraph"
```

### Step 2: Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate in PowerShell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy the template to create `.env` (do not commit `.env`!):
```powershell
Copy-Item .env.example .env
```
Fill in your credentials in `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_your_langsmith_api_key_here
LANGSMITH_PROJECT=General-AI-Chatbot
```

---

## 5. Pre-flight Testing

### 1. Test Groq Connectivity & LangSmith Tracing
Run the standalone pre-flight connection test:
```powershell
python test_groq.py
```
**Expected Output:**
```text
============================================================
  GROQ & LANGSMITH PRE-FLIGHT CONNECTION TEST
============================================================
[INFO] Groq API Key: gsk_... (Key loaded safely)
[INFO] Target Groq Model: openai/gpt-oss-120b
[INFO] LangSmith Tracing: true
[INFO] LangSmith Project: General-AI-Chatbot
...
============================================================
  GROQ LLM RESPONSE RECEIVED SUCCESSFULLY
============================================================
[SUCCESS] Groq API connectivity and authentication verified!
[SUCCESS] LangSmith automatic trace emitted to project: General-AI-Chatbot
```

### 2. Run Automated Unit & Agentic RAG Tests
Run the complete test suite (17 passed):
```powershell
python -m pytest tests -v
```
**Expected Output:**
```text
tests/test_agent.py::test_chatbot_node_success PASSED
tests/test_agent.py::test_chatbot_node_auth_error_handling PASSED
tests/test_agent.py::test_chatbot_node_rate_limit_error_handling PASSED
tests/test_agent.py::test_graph_structure PASSED
tests/test_config.py::test_get_llm_success PASSED
tests/test_config.py::test_get_llm_missing_key PASSED
tests/test_config.py::test_langsmith_config_presence PASSED
tests/test_memory.py::test_conversation_memory_multi_turn PASSED
tests/test_memory.py::test_thread_isolation PASSED
tests/test_market_research.py::test_knowledge_base_hybrid_search PASSED
tests/test_market_research.py::test_default_knowledge_base_preloaded PASSED
tests/test_market_research.py::test_tools_multi_source_retrieve PASSED
tests/test_market_research.py::test_plan_queries_node PASSED
tests/test_market_research.py::test_grade_documents_node PASSED
tests/test_market_research.py::test_decide_to_generate_conditional_routing PASSED
tests/test_market_research.py::test_market_research_graph_structure PASSED
tests/test_market_research.py::test_market_research_end_to_end_mocked PASSED

============================= 17 passed in 4.5s =============================
```

### 3. Test Standalone Graph Executions & SQLite Memory

**General AI Chatbot:**
```powershell
python agent.py
```

**Autonomous Market Research Agent (Agentic RAG):**
```powershell
python market_research/agent.py
```

---

## 6. Autonomous Market Research Agent (Agentic RAG)

Integrated directly into this repository is a specialized **Market Research Agent** implementing advanced **Agentic RAG Techniques**.

### Agentic RAG Architecture
```text
User Topic
    │
    ▼
[plan_queries] ────────► Decomposes topic into 3-4 sub-queries (TAM, Competitors, Drivers, Risks)
    │
    ▼
[retrieve] ────────────► Multi-source hybrid retrieval (BM25 + TF-IDF Knowledge Base + Wikipedia + Web)
    │
    ▼
[grade_documents] ─────► Self-Reflective Document Grading (Corrective RAG) filters noise
    │
    ├── (Insufficient evidence & under max iterations) ──► [rewrite_query] ──► [retrieve]
    │
    └── (Sufficient high-quality evidence)
          │
          ▼
[synthesize_report] ───► Generates executive-level C-Suite Market Research Report
          │
          ▼
[verify_grounding] ────► Cross-checks factual grounding against retrieved sources (anti-hallucination)
          │
          ▼
        [END]
```

### Key Capabilities:
1. **Dynamic Query Decomposition**: Splits broad market queries into focused research angles (Market Size & CAGR, Competitors, Technology Drivers, SWOT).
2. **Hybrid Multi-Source Retrieval**: Combines BM25 lexical search with TF-IDF semantic scoring over internal intelligence reports, supplemented by Wikipedia and live web tools.
3. **Self-Reflective Document Grading**: Filters out irrelevant noise and computes relevance scores before generation.
4. **Iterative Query Reformulation**: Broadens search strategies automatically if retrieved chunks are insufficient.
5. **Executive Framework Synthesis**: Automatically structures output with TAM/SAM/SOM sizing, competitor matrices, SWOT analysis, and GTM recommendations.
6. **Grounding & Source Verification**: Enforces factual adherence to retrieved evidence with detailed citation tracking.

### Interactive Web Application & Dual-Mode UI
Start the interactive server:
```powershell
python server.py
```
Open `http://localhost:8000` in your browser:
- **General Chat Mode**: ChatGPT-style multi-turn conversational AI with memory.
- **Market Research Mode**: 1-click toggle to invoke the Agentic RAG Market Research Agent with real-time status badges, pre-configured research topics, citation inspection, and verified grounding tags.

---

## 7. Running LangGraph Studio Locally

Start the local LangGraph development server:
```powershell
langgraph dev
```

1. The server will start on `http://127.0.0.1:2024`.
2. A browser tab will automatically open connecting to **LangGraph Studio**:
   `https://smith.langchain.com/studio/thread?baseUrl=http://127.0.0.1:2024`
3. In LangGraph Studio:
   - Select the `agent` graph.
   - Click **New Thread** or use the default thread.
   - Enter your prompt in the chat input.
   - Inspect the state, view the graph diagram, and see real-time streaming output.

---

## 7. Cloud Deployment Guide

> [!IMPORTANT]
> Never deploy without verifying your cloud account tier, permissions, and billing settings.

### Option A: LangGraph Cloud / LangSmith Deployments
1. **Prepare GitHub Repository**:
   ```powershell
   git status  # Ensure .env is NOT tracked
   git add .
   git commit -m "Prepare production LangGraph chatbot for deployment"
   ```
   Create a private repository on GitHub and push your code:
   ```powershell
   git remote add origin https://github.com/<your-username>/general-ai-chatbot-langgraph.git
   git branch -M main
   git push -u origin main
   ```
2. **Deploy on LangSmith**:
   - Go to [smith.langchain.com](https://smith.langchain.com) > **Deployments**.
   - Click **New Deployment**.
   - Connect your GitHub repository.
   - Select `langgraph.json` as the configuration file.
   - Add environment variables:
     - `GROQ_API_KEY`
     - `GROQ_MODEL=openai/gpt-oss-120b`
     - `LANGSMITH_TRACING=true`
     - `LANGSMITH_API_KEY`
     - `LANGSMITH_PROJECT=General-AI-Chatbot`
   - Click **Deploy**.

### Option B: Self-Hosted Docker Container
Build and run the production container:
```powershell
docker build -f deployment/Dockerfile -t general-ai-chatbot .
docker run -p 8000:8000 --env-file .env general-ai-chatbot
```

---

## 8. Security & Secret Management
- `.env` is listed in `.gitignore` and `.dockerignore`.
- API keys are never printed in plain text in logs or exception messages.
- Thread IDs provide complete multi-tenant conversation isolation.

---

## 9. Troubleshooting & FAQ
- **401 Unauthorized**: Check your `GROQ_API_KEY` in `.env`.
- **Model Not Found**: Verify that `GROQ_MODEL` is set to an active model on your Groq tier (e.g. `openai/gpt-oss-120b` or `qwen/qwen3.8-27b`). Use `python test_groq.py` to diagnose.
- **Port Conflict on 2024**: Run `langgraph dev --port 2025` if 2024 is in use.
