"""
Unit tests for the UI module.

This module contains unit tests for the Streamlit UI functionality,
including URL validation, article processing, and search.
"""

# Make sure pytest imports work correctly
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
import streamlit as st

from src.ui import validate_url, init_session_state, process_article_url, display_article_card


class TestUI:
    """Test cases for the UI module functions."""
    
    def test_validate_url_valid(self):
        """Test URL validation with valid URLs."""
        # Test with valid URLs
        assert validate_url("https://www.example.com") is True
        assert validate_url("http://example.com/path") is True
        assert validate_url("https://www.bbc.com/news/articles/c3r807j7xrwo") is True
    
    def test_validate_url_invalid(self):
        """Test URL validation with invalid URLs."""
        # Test with invalid URLs
        assert validate_url("example.com") is False  # missing scheme
        assert validate_url("https://") is False  # missing netloc
        assert validate_url("not a url") is False  # completely invalid
    
    @patch('src.ui.st.session_state')
    def test_init_session_state(self, mock_session_state):
        """Test session state initialization."""
        # Setup mock session state as dict-like
        mock_session_state.__contains__.side_effect = lambda key: key in {'search_results', 'processed_urls', 'related_searches'}
        
        # Call the function
        init_session_state()
        
        # Verify session state was initialized for missing keys
        mock_session_state.__setitem__.assert_any_call('search_results', [])
        mock_session_state.__setitem__.assert_any_call('processed_urls', [])
        mock_session_state.__setitem__.assert_any_call('related_searches', [])
    
    @patch('src.ui.st')
    @patch('src.ui.ArticleExtractor')
    @patch('src.ui.ArticleSummarizer')
    @patch('src.ui.st.session_state')
    def test_process_article_url_success(self, mock_session_state, mock_summarizer_class, mock_extractor_class, mock_st):
        """Test successful article processing."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract.return_value = MagicMock()
        
        mock_summarizer = MagicMock()
        mock_summarizer_class.return_value = mock_summarizer
        mock_summarizer.summarize.return_value = MagicMock()
        mock_summarizer.identify_topics.return_value = MagicMock()
        
        mock_db_service = MagicMock()
        mock_db_service.store_article.return_value = "test_doc_id"
        
        mock_session_state.processed_urls = []
        mock_session_state.db_service = mock_db_service
        
        # Call the function
        result = process_article_url("https://www.example.com")
        
        # Verify results
        assert result is True
        mock_extractor.extract.assert_called_once_with("https://www.example.com")
        mock_summarizer.summarize.assert_called_once()
        mock_summarizer.identify_topics.assert_called_once()
        mock_db_service.store_article.assert_called_once()
        assert "https://www.example.com" in mock_session_state.processed_urls
    
    @patch('src.ui.st')
    @patch('src.ui.ArticleExtractor')
    def test_process_article_url_failure(self, mock_extractor_class, mock_st):
        """Test article processing with exception handling."""
        # Setup mocks to raise exception
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract.side_effect = ValueError("Test error")
        
        # Call the function
        result = process_article_url("https://www.example.com")
        
        # Verify results
        assert result is False
        mock_extractor.extract.assert_called_once_with("https://www.example.com")
        mock_st.error.assert_called_once()
    
    @patch('src.ui.st')
    def test_display_article_card(self, mock_st):
        """Test article card display."""
        # Setup test article
        article = {
            'title': 'Test Article',
            'summary': 'This is a test summary',
            'url': 'https://www.example.com',
            'topics': ['technology', 'news'],
            'keywords': ['test', 'keyword'],
            'relevance_percentage': 85
        }
        
        # Mock st.container context manager
        mock_container = MagicMock()
        mock_st.container.return_value.__enter__.return_value = mock_container
        
        # Mock columns
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2]
        
        # Call the function
        display_article_card(article)
        
        # Verify article components are displayed
        mock_st.subheader.assert_called_with('Test Article')
        assert mock_st.markdown.call_count > 0
        assert mock_st.write.call_count > 0
