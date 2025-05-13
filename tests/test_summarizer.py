"""
Unit tests for the summarizer module.

This module contains tests for article summarization and topic identification.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pydantic import HttpUrl

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.summarizer import ArticleSummarizer
from src.models import ArticleContent, ArticleSummary, TopicIdentification


class TestArticleSummarizer:
    """Test cases for the ArticleSummarizer class."""

    @pytest.fixture
    def sample_article(self):
        """Fixture providing a sample article."""
        return ArticleContent(
            url=HttpUrl("https://www.example.com/news/article"),
            title="Test Article Title",
            text="This is a test article content. It contains information about various topics "
                 "such as technology, science, and politics. The article discusses recent "
                 "advancements in AI technology and its implications for society.",
            metadata={"author": "Test Author", "published_date": "2025-05-05"}
        )

    @pytest.fixture
    def mock_openai_env(self, monkeypatch):
        """Fixture to mock OpenAI environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-3.5-turbo")
        monkeypatch.setenv("OPENAI_TEMPERATURE", "0.5")
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "300")

    def test_summarize_concise(self, sample_article, mock_openai_env):
        """Test generating a concise summary."""
        with patch('src.summarizer.load_summarize_chain') as mock_chain:
            # Mock the summarize chain
            mock_chain_instance = MagicMock()
            mock_chain_instance.run.return_value = "This is a concise test summary."
            mock_chain.return_value = mock_chain_instance
            
            # Create summarizer and run summarization
            summarizer = ArticleSummarizer()
            result = summarizer.summarize(sample_article, summary_type="concise")
              # Verify the result
            assert isinstance(result, ArticleSummary)
            assert result.summary == "This is a concise test summary."
            assert result.summary_type == "concise"
            expected_id = str(hash(f"{sample_article.url}-{sample_article.title}"))
            assert result.article_id == expected_id

    def test_summarize_detailed(self, sample_article, mock_openai_env):
        """Test generating a detailed summary."""
        with patch('src.summarizer.load_summarize_chain') as mock_chain:
            # Mock the summarize chain
            mock_chain_instance = MagicMock()
            mock_chain_instance.run.return_value = "This is a detailed test summary with more information."
            mock_chain.return_value = mock_chain_instance
            
            # Create summarizer and run summarization
            summarizer = ArticleSummarizer()
            result = summarizer.summarize(sample_article, summary_type="detailed")
            
            # Verify the result            assert isinstance(result, ArticleSummary)
            assert result.summary == "This is a detailed test summary with more information."
            assert result.summary_type == "detailed"
            
    def test_identify_topics(self, sample_article, mock_openai_env):
        """Test identifying topics from an article."""
        with patch('src.summarizer.ChatOpenAI') as mock_chat:
            # Mock the chat model response
            mock_chat_instance = MagicMock()
            mock_message = MagicMock()
            mock_message.content = """```json
            {
                "topics": ["technology: Technology", "technology: Artificial Intelligence", "society: Society"],
                "keywords": ["technology: AI", "technology: advancements", "technology: technology", "society: society", "society: implications"]
            }
            ```"""
            mock_chat_instance.invoke.return_value = mock_message
            mock_chat.return_value = mock_chat_instance
            
            # Create summarizer and identify topics
            summarizer = ArticleSummarizer()
            result = summarizer.identify_topics(sample_article)
            
            # Verify the result
            assert isinstance(result, TopicIdentification)
            assert "technology: Technology" in result.topics
            assert "technology: Artificial Intelligence" in result.topics
            assert "technology: AI" in result.keywords
            assert "technology: technology" in result.keywords

    def test_summarize_error_handling(self, sample_article, mock_openai_env):
        """Test error handling during summarization."""
        with patch('src.summarizer.load_summarize_chain') as mock_chain:
            # Mock the summarize chain to raise an exception
            mock_chain.side_effect = Exception("Test error")
            
            # Create summarizer and expect error
            summarizer = ArticleSummarizer()
            with pytest.raises(ValueError) as excinfo:
                summarizer.summarize(sample_article)
                
            assert "Failed to summarize article" in str(excinfo.value)
            
    def test_parse_topics_response_valid_json(self, mock_openai_env):
        """Test parsing a valid JSON response for topics."""
        summarizer = ArticleSummarizer()
        response = """```json
        {
            "topics": ["politics: Politics", "economics: Economy", "politics: International Relations"],
            "keywords": ["politics: policy", "economics: finance", "international: global", "politics: treaty", "economics: trade"]
        }
        ```"""
        
        result = summarizer._parse_topics_response(response)
        
        assert "topics" in result
        assert "keywords" in result
        assert "politics: Politics" in result["topics"]
        assert "economics: finance" in result["keywords"]    def test_parse_topics_response_invalid_json(self, mock_openai_env):
        """Test parsing an invalid JSON response for topics."""
        summarizer = ArticleSummarizer()
        response = """
        Topics:
        - Technology
        - Science
        
        Keywords:
        - computer
        - research
        - innovation
        """
        
        result = summarizer._parse_topics_response(response)
        
        assert "topics" in result
        assert "keywords" in result
        assert len(result["topics"]) > 0
        assert len(result["keywords"]) > 0
        
        # Check that classifications were added
        assert all(":" in topic for topic in result["topics"])
        assert all(":" in keyword for keyword in result["keywords"])
