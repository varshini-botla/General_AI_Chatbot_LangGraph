"""
Verify deployed LangGraph cloud endpoint on Render.
Tests health check, thread creation, run execution, Groq LLM inference, and state retrieval.
"""

import json
import httpx

BASE_URL = "https://general-ai-chatbot-0gd3.onrender.com"

def test_deployed_endpoint():
    print("=" * 70)
    print("  VERIFYING DEPLOYED LANGGRAPH CLOUD ENDPOINT")
    print("  Target URL:", BASE_URL)
    print("=" * 70)

    # 1. Health check
    print("\n[STEP 1] Checking /info health check...")
    resp = httpx.get(f"{BASE_URL}/info", timeout=30.0)
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    info = resp.json()
    print(f"Server Version: {info.get('version')}")
    print(f"LangGraph Python Version: {info.get('langgraph_py_version')}")
    print("[SUCCESS] Cloud server is healthy and online!")

    # 2. Get Assistant ID
    print("\n[STEP 2] Discovering 'agent' assistant...")
    search_resp = httpx.post(f"{BASE_URL}/assistants/search", json={}, timeout=30.0)
    assert search_resp.status_code == 200
    assistants = search_resp.json()
    agent_assistant = next((a for a in assistants if a["name"] == "agent"), None)
    if not agent_assistant and isinstance(assistants, dict):
        agent_assistant = assistants
    assistant_id = agent_assistant["assistant_id"]
    print(f"Discovered Assistant ID: {assistant_id} (Graph: {agent_assistant.get('graph_id')})")

    # 3. Create a Cloud Thread
    print("\n[STEP 3] Creating a conversation thread in the cloud...")
    thread_resp = httpx.post(f"{BASE_URL}/threads", json={}, timeout=30.0)
    assert thread_resp.status_code == 200
    thread_data = thread_resp.json()
    thread_id = thread_data["thread_id"]
    print(f"Created Thread ID: {thread_id}")

    # 4. Invoke the Deployed Chatbot
    test_question = "What is Artificial Intelligence? Answer in 2 clear sentences."
    print(f"\n[STEP 4] Sending live query to deployed agent:")
    print(f"User: '{test_question}'")
    
    payload = {
        "assistant_id": assistant_id,
        "input": {
            "messages": [
                {"role": "user", "content": test_question}
            ]
        }
    }
    
    run_resp = httpx.post(
        f"{BASE_URL}/threads/{thread_id}/runs/wait",
        json=payload,
        timeout=60.0
    )
    assert run_resp.status_code == 200, f"Run failed: {run_resp.status_code} - {run_resp.text}"
    run_result = run_resp.json()
    
    # 5. Retrieve Thread State to see AI Response
    state_resp = httpx.get(f"{BASE_URL}/threads/{thread_id}/state", timeout=30.0)
    assert state_resp.status_code == 200
    state_data = state_resp.json()
    messages = state_data["values"]["messages"]
    
    ai_message = messages[-1]
    print("\n" + "=" * 70)
    print("  LIVE GROQ RESPONSE FROM DEPLOYED CLOUD SERVER")
    print("=" * 70)
    print(f"\n{ai_message['content']}\n")
    print("=" * 70)
    print("[SUCCESS] Cloud deployment verification complete!")
    print("[SUCCESS] Deployed URL:", BASE_URL)
    print("=" * 70)

if __name__ == "__main__":
    test_deployed_endpoint()
