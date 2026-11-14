# Production Dockerfile for General-Purpose LangGraph AI Chatbot
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY agent.py langgraph.json ./
COPY data/ ./data/

EXPOSE 8000

# Start LangGraph Agent Server dynamically using platform PORT
CMD ["sh", "-c", "langgraph dev --host 0.0.0.0 --port ${PORT:-8000} --no-browser"]
