"""
Groq API and LangSmith Connection Test Script
Tests connectivity to Groq LLM and verifies LangSmith tracing environment.
"""

import os
import sys
from dotenv import load_dotenv

# 1. Load environment variables from .env
load_dotenv()

def mask_key(key: str | None) -> str:
    """Safely mask API key for logging without exposing secret characters."""
    if not key:
        return "<NOT SET>"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"

def test_groq_connection():
    print("=" * 60)
    print("  GROQ & LANGSMITH PRE-FLIGHT CONNECTION TEST")
    print("=" * 60)

    # 2. Verify Groq API Key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("[ERROR] GROQ_API_KEY is not set in your .env file.")
        print("Please check your .env configuration.")
        sys.exit(1)

    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    print(f"[INFO] Groq API Key: {mask_key(groq_api_key)} (Key loaded safely)")
    print(f"[INFO] Target Groq Model: {groq_model}")

    # 3. Check LangSmith Configuration
    langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false")
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    langsmith_project = os.getenv("LANGSMITH_PROJECT", "General-AI-Chatbot")
    print(f"[INFO] LangSmith Tracing: {langsmith_tracing}")
    print(f"[INFO] LangSmith Project: {langsmith_project}")
    print(f"[INFO] LangSmith API Key: {mask_key(langsmith_api_key)}")

    # 4. Initialize Groq LLM Client
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        print("[ERROR] langchain-groq package is not installed. Please run: pip install -r requirements.txt")
        sys.exit(1)

    print("\n[STEP 1] Initializing ChatGroq client...")
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=groq_model,
            temperature=0.7,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize ChatGroq: {type(e).__name__}: {e}")
        sys.exit(1)

    # 5. Send Real Test Prompt
    test_prompt = "Hello! In one concise sentence, what is your role as a general AI assistant?"
    print(f"[STEP 2] Sending test prompt to Groq: '{test_prompt}'")

    try:
        response = llm.invoke(test_prompt)
        print("\n" + "=" * 60)
        print("  GROQ LLM RESPONSE RECEIVED SUCCESSFULLY")
        print("=" * 60)
        print(f"\n{response.content}\n")
        print("=" * 60)
        print("[SUCCESS] Groq API connectivity and authentication verified!")
        if langsmith_tracing.lower() == "true":
            print("[SUCCESS] LangSmith automatic trace emitted to project:", langsmith_project)
        print("=" * 60)
    except Exception as e:
        err_msg = str(e)
        print("\n[ERROR] An error occurred while communicating with Groq:")
        if "AuthenticationError" in type(e).__name__ or "401" in err_msg:
            print(" -> Authentication failed: The GROQ_API_KEY provided is invalid or unauthorized.")
        elif "NotFoundError" in type(e).__name__ or "model_not_found" in err_msg:
            print(f" -> Model not found: '{groq_model}' may not be supported or available on your Groq tier.")
        elif "RateLimitError" in type(e).__name__ or "429" in err_msg:
            print(" -> Rate limit exceeded on Groq API.")
        else:
            print(f" -> {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_groq_connection()
