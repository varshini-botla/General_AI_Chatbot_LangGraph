"""
Comprehensive Unit & Integration Test Suite for Agentic RAG Market Research Agent.
Tests:
1. Knowledge Base indexing and hybrid search (BM25 + TF-IDF)
2. Tool multi-source retrieval & deduplication
3. Query planning & decomposition node
4. Self-reflective document grading node
5. Conditional routing logic (decide_to_generate)
6. StateGraph compilation and edge integrity
7. End-to-end workflow execution with SQLite memory
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from market_research.knowledge_base import MarketKnowledgeBase, get_default_knowledge_base
from market_research.tools import MarketResearchTools
from market_research.agent import (
    plan_queries_node,
    retrieve_node,
    grade_documents_node,
    decide_to_generate,
    rewrite_query_node,
    synthesize_report_node,
    verify_grounding_node,
    create_market_research_graph,
    MarketResearchState,
)
from langgraph.checkpoint.memory import MemorySaver


def test_knowledge_base_hybrid_search():
    """Test BM25 + TF-IDF hybrid retrieval on indexed market documents."""
    kb = MarketKnowledgeBase()
    kb.add_document(
        title="Enterprise AI Market Analysis",
        content="Enterprise AI software is experiencing massive growth in autonomous agent workflows.",
        category="AI",
    )
    kb.add_document(
        title="Solar Power Energy Report",
        content="Photovoltaic solar panels and utility battery storage installations reached records.",
        category="CleanTech",
    )
    kb.build_index()

    # Query for AI
    ai_results = kb.hybrid_search("autonomous agent workflows", top_k=2)
    assert len(ai_results) > 0
    assert "Enterprise AI" in ai_results[0]["title"]
    assert ai_results[0]["score"] > 0

    # Query for Solar
    solar_results = kb.hybrid_search("solar panels battery storage", top_k=2)
    assert len(solar_results) > 0
    assert "Solar Power" in solar_results[0]["title"]


def test_default_knowledge_base_preloaded():
    """Test that default market knowledge base is pre-populated with diverse sectors."""
    kb = get_default_knowledge_base()
    assert len(kb.documents) >= 5

    categories = {d["category"] for d in kb.documents}
    assert "Artificial Intelligence" in categories
    assert "Semiconductors & Hardware" in categories

    search_res = kb.hybrid_search("Groq LPU NVIDIA GPU inference", top_k=1)
    assert len(search_res) == 1
    assert "Cloud AI Semiconductor" in search_res[0]["title"]


def test_tools_multi_source_retrieve():
    """Test MarketResearchTools queries multiple sources and removes duplicates."""
    tools = MarketResearchTools()
    sub_queries = ["AI Agents market size", "Specialized AI chips Groq NVIDIA"]

    with patch.object(tools, "search_wikipedia", return_value=[]), \
         patch.object(tools, "search_web", return_value=[]):
        docs = tools.multi_source_retrieve(sub_queries, top_k_per_query=2)
        assert len(docs) >= 1
        # Check source type is properly tagged
        assert docs[0]["source_type"] == "Internal Market Intelligence"


def test_plan_queries_node():
    """Test query planning node generates structured sub-queries."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='["AI market size CAGR", "Top competitors in agentic AI", "Enterprise adoption drivers"]'
    )

    with patch("market_research.agent.get_llm", return_value=mock_llm):
        state: MarketResearchState = {
            "messages": [HumanMessage(content="AI Agents")],
            "research_topic": "AI Agents",
            "sub_queries": [],
            "retrieved_docs": [],
            "graded_docs": [],
            "iteration": 0,
            "max_iterations": 2,
            "synthesis_report": "",
            "grounding_status": "",
            "citations": [],
        }
        res = plan_queries_node(state)
        assert len(res["sub_queries"]) == 3
        assert "AI market size CAGR" in res["sub_queries"]
        assert res["iteration"] == 1


def test_grade_documents_node():
    """Test self-reflective document grading filters out noise."""
    state: MarketResearchState = {
        "messages": [],
        "research_topic": "AI Agents",
        "sub_queries": ["AI Agents market"],
        "retrieved_docs": [
            {
                "title": "Global AI Agents Market",
                "source_type": "Internal Market Intelligence",
                "content": "The global AI Agents market size was estimated at USD 5.1 billion in 2024 with a 44.8% CAGR.",
                "score": 0.85,
            },
            {
                "title": "Irrelevant noise snippet",
                "source_type": "Web",
                "content": "Short text",  # Below threshold
                "score": 0.01,
            },
        ],
        "graded_docs": [],
        "iteration": 1,
        "max_iterations": 2,
        "synthesis_report": "",
        "grounding_status": "",
        "citations": [],
    }

    res = grade_documents_node(state)
    assert len(res["graded_docs"]) == 1
    assert res["graded_docs"][0]["title"] == "Global AI Agents Market"
    assert len(res["citations"]) == 1


def test_decide_to_generate_conditional_routing():
    """Test conditional edge routes to synthesize_report or rewrite_query."""
    # Case 1: Sufficient evidence -> synthesize
    state_sufficient: MarketResearchState = {
        "graded_docs": [{"doc": 1}, {"doc": 2}],
        "iteration": 1,
        "max_iterations": 2,
    }  # type: ignore
    assert decide_to_generate(state_sufficient) == "synthesize_report"

    # Case 2: Insufficient evidence & under max iterations -> rewrite
    state_insufficient: MarketResearchState = {
        "graded_docs": [{"doc": 1}],
        "iteration": 1,
        "max_iterations": 2,
    }  # type: ignore
    assert decide_to_generate(state_insufficient) == "rewrite_query"

    # Case 3: Reached max iterations -> force synthesis
    state_max_iter: MarketResearchState = {
        "graded_docs": [{"doc": 1}],
        "iteration": 2,
        "max_iterations": 2,
    }  # type: ignore
    assert decide_to_generate(state_max_iter) == "synthesize_report"


def test_market_research_graph_structure():
    """Test LangGraph StateGraph nodes, edges, and compilation."""
    compiled = create_market_research_graph()
    assert compiled is not None
    expected_nodes = {
        "plan_queries",
        "retrieve",
        "grade_documents",
        "rewrite_query",
        "synthesize_report",
        "verify_grounding",
    }
    for node_name in expected_nodes:
        assert node_name in compiled.nodes


def test_market_research_end_to_end_mocked():
    """Test full execution of the Agentic RAG graph with MemorySaver checkpointer."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        # 1. plan_queries_node
        AIMessage(content='["Autonomous AI Agents market size", "Key competitors in AI agents"]'),
        # 2. synthesize_report_node
        AIMessage(content="# Executive Market Report: AI Agents\n\n## 1. Market Sizing\nMarket projected to grow at 44.8% CAGR."),
    ]

    with patch("market_research.agent.get_llm", return_value=mock_llm):
        checkpointer = MemorySaver()
        test_graph = create_market_research_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test_mr_thread"}}

        init_state = {
            "messages": [HumanMessage(content="AI Agents in Enterprise Software")],
            "research_topic": "AI Agents in Enterprise Software",
            "sub_queries": [],
            "retrieved_docs": [],
            "graded_docs": [],
            "iteration": 0,
            "max_iterations": 2,
            "synthesis_report": "",
            "grounding_status": "",
            "citations": [],
        }

        result = test_graph.invoke(init_state, config=config)

        assert "messages" in result
        assert len(result["messages"]) == 2  # 1 User + 1 AI
        final_text = result["messages"][-1].content
        assert "Market Report" in final_text
        assert "Market Sizing" in final_text
        assert result.get("grounding_status") == "Verified Grounded in Retrieved Multi-Source Evidence"
