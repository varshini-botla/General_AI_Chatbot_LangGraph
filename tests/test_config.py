"""
Unit tests for Chatbot Configuration and Environment Validation.
"""

import os
import pytest
from unittest.mock import patch
from agent import get_llm


def test_get_llm_success():
    """Test get_llm initializes successfully when GROQ_API_KEY is present."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_dummy_test_key_12345", "GROQ_MODEL": "openai/gpt-oss-120b"}):
        llm = get_llm()
        assert llm is not None
        assert llm.model_name == "openai/gpt-oss-120b"


def test_get_llm_missing_key():
    """Test get_llm raises ValueError when GROQ_API_KEY is missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
            get_llm()


def test_langsmith_config_presence():
    """Test that LangSmith environment variables are configured."""
    tracing = os.getenv("LANGSMITH_TRACING")
    project = os.getenv("LANGSMITH_PROJECT")
    assert tracing is not None
    assert project == "General-AI-Chatbot"
