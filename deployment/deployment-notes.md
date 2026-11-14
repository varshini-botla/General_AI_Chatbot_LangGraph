# Cloud Deployment Architecture & Notes

## 1. Overview
The **General-Purpose AI Chatbot** uses the same LangGraph `agent:graph` architecture both locally and in cloud production.

Local Architecture:
```text
VS Code / CLI -> langgraph dev -> LangGraph Studio -> Chatbot Node (agent.py) -> Groq API -> Response
                                                   |
                                                   v
                                        LangSmith Tracing (General-AI-Chatbot)
```

Cloud Architecture:
```text
Client / Web UI -> Hosted LangGraph Agent Server -> Chatbot Node (agent.py) -> Groq API -> Response
                                                   |
                                                   +--> Managed Checkpointer (Postgres / Cloud Store)
                                                   |
                                                   +--> LangSmith Production Traces
```

---

## 2. Platform Options

### Option A: LangSmith Cloud / LangGraph Cloud (Managed)
- **Official Docs**: `https://docs.langchain.com/langsmith/deployments`
- **Features**: Fully managed serverless LangGraph runtime, automated persistent state, zero infrastructure overhead.
- **Requirements**:
  1. LangSmith Account with Deployment tier enabled (Cloud Plus / Enterprise or active deployment beta quota).
  2. GitHub integration: The repository is pushed to a private/public GitHub repo and connected via LangSmith UI or `langgraph deploy`.
  3. Environment variables configured in LangSmith UI (`GROQ_API_KEY`, `GROQ_MODEL`, `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT=General-AI-Chatbot`).

### Option B: Self-Hosted Container (Docker / Render / Railway / GCP Cloud Run)
- If your LangSmith tier is free Developer tier without Cloud Deployments, you can host the LangGraph API server container:
  - Uses the included [`Dockerfile`](file:///C:/projects/AI%20CHATBOT/General_AI_Chatbot_LangGraph/deployment/Dockerfile).
  - In production, configure durable state persistence using PostgreSQL (`langgraph-checkpoint-postgres`) rather than local ephemeral SQLite.
  - Exposes port 8000 with HTTPS enforced by the hosting provider.

---

## 3. Production Environment Variables
Configure these in the Cloud Platform Settings / Secrets Manager (never commit `.env`!):

| Variable | Description | Example / Recommended Value |
|---|---|---|
| `GROQ_API_KEY` | Groq API Key | `gsk_...` |
| `GROQ_MODEL` | Target Groq LLM | `openai/gpt-oss-120b` or `qwen/qwen3.8-27b` |
| `LANGSMITH_TRACING` | Enables automatic tracing | `true` |
| `LANGSMITH_API_KEY` | LangSmith API Key | `lsv2_...` |
| `LANGSMITH_PROJECT` | Project name in LangSmith | `General-AI-Chatbot` |

---

## 4. Deployment Verification Checklist
- [x] Compiled graph exported in `agent.py` as `graph`
- [x] Valid `langgraph.json` declaring `./agent.py:graph`
- [x] All 9 automated tests passing (`pytest tests/`)
- [x] Live Groq model response verified (`python test_groq.py`)
- [x] Secrets excluded from Git repository (`.gitignore` verified)
- [ ] Explicit user authorization obtained before publishing/deploying to GitHub/Cloud.
