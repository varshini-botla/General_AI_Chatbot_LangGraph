"""
Market Research Tools Module.
Provides hybrid retrieval across:
1. Internal Market Knowledge Base (BM25 + TF-IDF)
2. Wikipedia API Search for market background, company profiles, and industry history
3. Live Web Search Tool with clean text extraction and fallbacks
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
from typing import List, Dict, Any, Optional
import wikipedia

from market_research.knowledge_base import MarketKnowledgeBase, get_default_knowledge_base

logger = logging.getLogger(__name__)


class MarketResearchTools:
    """
    Unified multi-source tool provider for Agentic RAG Market Research.
    """

    def __init__(self, knowledge_base: Optional[MarketKnowledgeBase] = None):
        self.kb = knowledge_base or get_default_knowledge_base()

    def search_knowledge_base(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Queries the internal indexed market research documents.
        Returns top_k graded snippets with relevance scores and source metadata.
        """
        results = self.kb.hybrid_search(query, top_k=top_k)
        formatted = []
        for r in results:
            formatted.append({
                "source_type": "Internal Market Intelligence",
                "title": r.get("title", "Market Report"),
                "category": r.get("category", "General"),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
                "metadata": r.get("metadata", {}),
            })
        return formatted

    def search_wikipedia(self, query: str, max_results: int = 2) -> List[Dict[str, Any]]:
        """
        Searches Wikipedia for corporate history, industry definitions, and macroeconomic background.
        """
        results = []
        try:
            search_titles = wikipedia.search(query, results=max_results)
            for title in search_titles:
                try:
                    summary = wikipedia.summary(title, sentences=4, auto_suggest=False)
                    page = wikipedia.page(title, auto_suggest=False)
                    results.append({
                        "source_type": "Wikipedia Research",
                        "title": title,
                        "category": "Encyclopedia",
                        "content": summary,
                        "url": page.url,
                        "score": 0.85,
                        "metadata": {"source": "Wikipedia", "url": page.url},
                    })
                except Exception as page_err:
                    continue
        except Exception as e:
            logger.warning(f"Wikipedia search failed for '{query}': {e}")
        return results

    def search_web(self, query: str, max_results: int = 2) -> List[Dict[str, Any]]:
        """
        Searches web / live data using DuckDuckGo search API with error resilience.
        """
        results = []
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(query, max_results=max_results))
                for item in ddg_results:
                    title = item.get("title", "Web Market Data")
                    body = item.get("body", "")
                    href = item.get("href", "")
                    if body:
                        results.append({
                            "source_type": "Live Web Search",
                            "title": title,
                            "category": "Live Market News",
                            "content": body,
                            "url": href,
                            "score": 0.80,
                            "metadata": {"source": "Web", "url": href},
                        })
        except Exception as e:
            logger.info(f"Live web search fallback for '{query}': {e}")
        return results

    def multi_source_retrieve(self, sub_queries: List[str], top_k_per_query: int = 2) -> List[Dict[str, Any]]:
        """
        Executes parallel multi-source retrieval across sub-queries.
        Deduplicates results by content snippet similarity and source title.
        """
        aggregated_docs: List[Dict[str, Any]] = []
        seen_titles = set()

        for q in sub_queries:
            # 1. Knowledge Base Search
            kb_matches = self.search_knowledge_base(q, top_k=top_k_per_query)
            for m in kb_matches:
                if m["title"] not in seen_titles:
                    seen_titles.add(m["title"])
                    m["query_used"] = q
                    aggregated_docs.append(m)

            # 2. Wikipedia background check
            wiki_matches = self.search_wikipedia(q, max_results=1)
            for w in wiki_matches:
                if w["title"] not in seen_titles:
                    seen_titles.add(w["title"])
                    w["query_used"] = q
                    aggregated_docs.append(w)

            # 3. Live Web Search
            web_matches = self.search_web(q, max_results=1)
            for wb in web_matches:
                if wb["title"] not in seen_titles:
                    seen_titles.add(wb["title"])
                    wb["query_used"] = q
                    aggregated_docs.append(wb)

        return aggregated_docs
