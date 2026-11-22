"""
Market Research Agent Package with Agentic RAG Techniques.
Provides intelligent query planning, hybrid multi-source retrieval,
self-reflective document grading, iterative query rewriting,
and structured executive market report generation with grounding verification.
"""

from market_research.knowledge_base import MarketKnowledgeBase, get_default_knowledge_base
from market_research.tools import MarketResearchTools
from market_research.agent import (
    create_market_research_graph,
    graph,
    MarketResearchState,
)

__all__ = [
    "MarketKnowledgeBase",
    "get_default_knowledge_base",
    "MarketResearchTools",
    "create_market_research_graph",
    "graph",
    "MarketResearchState",
]
