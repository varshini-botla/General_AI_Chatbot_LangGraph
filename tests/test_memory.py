"""
Unit tests for Conversation Memory, Thread Isolation, and Checkpointing.
"""

from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from agent import create_graph


def test_conversation_memory_multi_turn():
    """Test that multiple turns in the same thread preserve conversation history."""
    mock_llm = MagicMock()
    # First turn response, second turn response
    mock_llm.invoke.side_effect = [
        AIMessage(content="Machine Learning is a subset of AI."),
        AIMessage(content="Supervised, unsupervised, and reinforcement learning."),
    ]
    
    with patch("agent.get_llm", return_value=mock_llm):
        checkpointer = MemorySaver()
        test_graph = create_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test_thread_1"}}
        
        # Turn 1
        res1 = test_graph.invoke(
            {"messages": [HumanMessage(content="Explain Machine Learning")]},
            config=config,
        )
        assert len(res1["messages"]) == 2  # 1 User + 1 AI
        assert res1["messages"][-1].content == "Machine Learning is a subset of AI."
        
        # Turn 2 (Follow-up)
        res2 = test_graph.invoke(
            {"messages": [HumanMessage(content="What are its types?")]},
            config=config,
        )
        assert len(res2["messages"]) == 4  # 2 User + 2 AI
        assert res2["messages"][-1].content == "Supervised, unsupervised, and reinforcement learning."


def test_thread_isolation():
    """Test that two different threads have completely isolated conversation states."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content="Thread A response about Python."),
        AIMessage(content="Thread B response about Java."),
    ]
    
    with patch("agent.get_llm", return_value=mock_llm):
        checkpointer = MemorySaver()
        test_graph = create_graph(checkpointer=checkpointer)
        
        config_a = {"configurable": {"thread_id": "thread_A"}}
        config_b = {"configurable": {"thread_id": "thread_B"}}
        
        # Invoke Thread A
        res_a = test_graph.invoke(
            {"messages": [HumanMessage(content="Python topic")]},
            config=config_a,
        )
        
        # Invoke Thread B
        res_b = test_graph.invoke(
            {"messages": [HumanMessage(content="Java topic")]},
            config=config_b,
        )
        
        # Verify thread A only has 2 messages
        assert len(res_a["messages"]) == 2
        assert res_a["messages"][-1].content == "Thread A response about Python."
        
        # Verify thread B only has 2 messages and does not contain thread A messages
        assert len(res_b["messages"]) == 2
        assert res_b["messages"][-1].content == "Thread B response about Java."
        assert "Python" not in res_b["messages"][0].content
