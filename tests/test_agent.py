"""
Unit tests for LangGraph Agent Graph, State, and Chatbot Node.
"""

from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent import chatbot_node, create_graph, graph, ChatState, SYSTEM_PROMPT


def test_chatbot_node_success():
    """Test chatbot_node appends AI response when LLM responds successfully."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Hello! How can I help you today?")
    
    with patch("agent.get_llm", return_value=mock_llm):
        state: ChatState = {"messages": [HumanMessage(content="Hello")]}
        result = chatbot_node(state)
        
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Hello! How can I help you today?"
        
        # Verify system prompt was prepended to the prompt passed to LLM
        called_args = mock_llm.invoke.call_args[0][0]
        assert isinstance(called_args[0], SystemMessage)
        assert "General-Purpose AI Assistant" in called_args[0].content


def test_chatbot_node_auth_error_handling():
    """Test chatbot_node handles authentication errors gracefully without leaking keys."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("401 Unauthorized: Invalid API key")
    
    with patch("agent.get_llm", return_value=mock_llm):
        state: ChatState = {"messages": [HumanMessage(content="Hello")]}
        result = chatbot_node(state)
        
        assert "messages" in result
        ai_resp = result["messages"][0].content
        assert "authentication error" in ai_resp.lower()
        # Ensure no private key is leaked
        assert "gsk_" not in ai_resp


def test_chatbot_node_rate_limit_error_handling():
    """Test chatbot_node handles rate limits with informative feedback."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("429 RateLimitError: Rate limit reached")
    
    with patch("agent.get_llm", return_value=mock_llm):
        state: ChatState = {"messages": [HumanMessage(content="Hello")]}
        result = chatbot_node(state)
        
        ai_resp = result["messages"][0].content
        assert "rate limit" in ai_resp.lower()


def test_graph_structure():
    """Test LangGraph StateGraph nodes and edges."""
    compiled = create_graph()
    assert compiled is not None
    # Verify graph contains the chatbot node
    assert "chatbot" in compiled.nodes
