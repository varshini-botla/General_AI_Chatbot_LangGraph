"""
Market Knowledge Base and Hybrid Retriever (BM25 + TF-IDF).
Stores curated industry market research documents and indexes user-provided documents.
Executes hybrid ranking combining keyword (BM25) and semantic lexical (TF-IDF) scores.
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


class MarketKnowledgeBase:
    """
    Self-contained, in-memory Market Knowledge Base with Hybrid Search (BM25 + TF-IDF).
    Zero external database dependencies or cloud billing required.
    """

    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._tokenized_corpus: List[List[str]] = []
        self._is_indexed = False

    def add_document(self, title: str, content: str, category: str, metadata: Optional[Dict[str, Any]] = None):
        """Adds a document to the knowledge base."""
        doc = {
            "id": f"doc_{len(self.documents) + 1}",
            "title": title,
            "content": content,
            "category": category,
            "metadata": metadata or {},
        }
        self.documents.append(doc)
        self._is_indexed = False

    def add_documents(self, docs: List[Dict[str, Any]]):
        """Batch add documents."""
        for d in docs:
            self.add_document(
                title=d.get("title", "Untitled Report"),
                content=d.get("content", ""),
                category=d.get("category", "General"),
                metadata=d.get("metadata", {}),
            )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple alphanumeric tokenizer for BM25."""
        return re.findall(r"\b\w+\b", text.lower())

    def build_index(self):
        """Builds BM25 index and TF-IDF matrix over indexed documents."""
        if not self.documents:
            return

        corpus_texts = [f"{d['title']}\n{d['content']}" for d in self.documents]
        self._tokenized_corpus = [self._tokenize(txt) for txt in corpus_texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        self._tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus_texts)
        self._is_indexed = True

    def hybrid_search(self, query: str, top_k: int = 4, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining BM25 and TF-IDF cosine similarity.
        alpha: weight for BM25 score (1 - alpha for TF-IDF cosine score).
        Returns top_k most relevant document chunks with scores.
        """
        if not self._is_indexed:
            self.build_index()

        if not self.documents or not self._bm25 or self._tfidf_matrix is None:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # 1. BM25 Scores
        bm25_scores = np.array(self._bm25.get_scores(query_tokens))
        bm25_max = np.max(bm25_scores) if np.max(bm25_scores) > 0 else 1.0
        norm_bm25 = bm25_scores / bm25_max

        # 2. TF-IDF Cosine Similarity
        query_vec = self._tfidf_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        tfidf_max = np.max(tfidf_scores) if np.max(tfidf_scores) > 0 else 1.0
        norm_tfidf = tfidf_scores / tfidf_max

        # 3. Hybrid Score Combination
        combined_scores = alpha * norm_bm25 + (1.0 - alpha) * norm_tfidf

        # Sort indices descending
        ranked_indices = np.argsort(combined_scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            score = float(combined_scores[idx])
            if score > 0.05:  # Relevance threshold
                doc = self.documents[idx].copy()
                doc["score"] = round(score, 4)
                doc["bm25_score"] = round(float(norm_bm25[idx]), 4)
                doc["tfidf_score"] = round(float(norm_tfidf[idx]), 4)
                results.append(doc)

        return results


# Built-in High-Value Market Intelligence Knowledge Base
def get_default_knowledge_base() -> MarketKnowledgeBase:
    """Instantiates and pre-populates the default industry market intelligence knowledge base."""
    kb = MarketKnowledgeBase()

    curated_reports = [
        {
            "title": "Global AI Agents and Autonomous Workflow Market (2025-2030)",
            "category": "Artificial Intelligence",
            "content": (
                "The global AI Agents market size was estimated at USD 5.1 billion in 2024 and is projected "
                "to expand at a compound annual growth rate (CAGR) of 44.8% from 2025 to 2030, reaching USD 48.6 billion. "
                "Key drivers include enterprise demand for multi-agent orchestration, automated code development, "
                "customer support automation, and Agentic RAG workflows in knowledge management. "
                "Major industry leaders include OpenAI (Operator, Swarm), Anthropic (Claude Computer Use), Microsoft "
                "(Copilot Studio, AutoGen), Google (Vertex AI Agents), LangChain (LangGraph), and CrewAI. "
                "Primary challenges include agent hallucination risks, enterprise security/RBAC integration, "
                "high token latency, and multi-tool reliability. TAM is forecasted to exceed $100B by 2032."
            ),
            "metadata": {"source": "Global Tech Intelligence Index", "year": 2025, "sector": "Enterprise AI"},
        },
        {
            "title": "Cloud AI Semiconductor & Specialized Inference Chips (Groq, NVIDIA, TPU)",
            "category": "Semiconductors & Hardware",
            "content": (
                "The AI Accelerator and Inference Hardware market reached $53 billion in 2024, propelled by surging LLM inference "
                "workloads. NVIDIA dominates datacenter training with 82% market share (Blackwell B200 and H100 GPUs). "
                "However, specialized LPU (Language Processing Unit) architectures from Groq, Cerebras, and Google TPUs are "
                "capturing high-velocity token serving market share due to deterministic low latency (300-800 tokens/sec) "
                "and cost-per-million-token advantages. Hyperscalers (AWS Trainium/Inferentia, Azure Maia) are shifting 25% "
                "of capital expenditures to custom silicon to reduce NVIDIA reliance. Supply chain constraints on HBM3e "
                "memory and TSMC packaging capacity remain key bottleneck risks through 2026."
            ),
            "metadata": {"source": "Semiconductor Market Review", "year": 2025, "sector": "Hardware"},
        },
        {
            "title": "Enterprise Retrieval-Augmented Generation (RAG) & Vector Database Industry Analysis",
            "category": "Software Infrastructure",
            "content": (
                "The Enterprise RAG and Search infrastructure market is expected to grow from $1.8 billion in 2024 "
                "to $11.4 billion by 2029 (CAGR 45.2%). Standard naive vector search is rapidly evolving into Agentic RAG, "
                "incorporating query decomposition, document relevance self-grading, graph-based knowledge retrieval (GraphRAG), "
                "and multi-hop reasoning. Leading vector database vendors include Pinecone, Qdrant, Milvus (Zilliz), "
                "Weaviate, and embedded engines like SQLite-vec and pgvector. Enterprise buyer surveys show 68% of Fortune 500 "
                "companies are deploying RAG pipelines for internal documentation, regulatory compliance, and support desks."
            ),
            "metadata": {"source": "Enterprise Data Infrastructure Benchmark", "year": 2025, "sector": "Data Architecture"},
        },
        {
            "title": "SaaS Platform Evolution: Agentic Workflow Automation vs Legacy RPA",
            "category": "Enterprise Software",
            "content": (
                "Traditional Robotic Process Automation (RPA) vendors like UiPath and Automation Anywhere are being disrupted "
                "by LLM-native agentic platforms. Enterprise software seat-based pricing is shifting toward consumption-based "
                "and outcome-based pricing models. Key players: Salesforce (Agentforce), ServiceNow (Now Assist), "
                "and Workday are embedding autonomous reasoning agents directly into CRM and ERP workflows. "
                "Adoption barriers include auditability, error correction, and executive trust. The total addressable "
                "market (TAM) for enterprise automation is estimated at $370 billion globally."
            ),
            "metadata": {"source": "SaaS Executive Monitor", "year": 2025, "sector": "Enterprise SaaS"},
        },
        {
            "title": "Global Electric Vehicle (EV) and Autonomous Driving Technology Landscape",
            "category": "Automotive & Mobility",
            "content": (
                "Global electric vehicle sales reached 17.5 million units in 2024, representing over 21% of total automotive "
                "sales worldwide. China remains the largest market (over 60% of global EV production), led by BYD, Geely, "
                "and Tesla. In autonomous driving, Level 2+ / Level 3 supervised autonomy is becoming standard in premium tiers. "
                "Robotaxi commercial operations led by Waymo have surpassed 100,000 weekly paid rides in US metros, "
                "while Tesla's Full Self-Driving (FSD) transitioned to end-to-end neural network architectures. "
                "Key market challenges include trade tariffs, public charging infrastructure reliability, and battery raw material costs."
            ),
            "metadata": {"source": "Mobility & CleanTech Quarterly", "year": 2025, "sector": "Automotive"},
        },
        {
            "title": "Cybersecurity & AI-Driven Threat Detection Market Forecast",
            "category": "Cybersecurity",
            "content": (
                "The cybersecurity software and managed detection market is forecasted to expand from $190 billion in 2024 "
                "to $300 billion by 2030 (CAGR 10.6%). The weaponization of generative AI for phishing and polymorphic malware "
                "has accelerated enterprise spending on AI-native Security Operations Centers (SOC). Top vendors include "
                "CrowdStrike (Falcon platform), Palo Alto Networks (Cortex XSIAM), Microsoft Defender, and Cloudflare. "
                "Zero Trust Architecture (ZTA) and automated incident response agents are the primary investment priorities."
            ),
            "metadata": {"source": "Cyber Defense Annual Outlook", "year": 2025, "sector": "Cybersecurity"},
        },
        {
            "title": "FinTech: Embedded Finance, Next-Gen Payments, and AI Underwriting",
            "category": "Financial Technology",
            "content": (
                "The Embedded Finance market is projected to reach $248 billion by 2032. Financial institutions and fintechs "
                "are replacing legacy rule-based credit scoring with real-time AI underwriting models that incorporate alternative "
                "transaction data. Payment giants Stripe, Adyen, and PayPal are expanding into unified banking-as-a-service (BaaS). "
                "Regulatory scrutiny (Open Banking standards, Consumer Financial Protection rules) and fraud mitigation "
                "are the central compliance challenges for market entrants."
            ),
            "metadata": {"source": "Global FinTech Trends", "year": 2025, "sector": "Finance"},
        },
    ]

    kb.add_documents(curated_reports)
    kb.build_index()
    return kb
