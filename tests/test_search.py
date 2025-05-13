"""
Unit tests for the semantic search module.

This module contains unit tests for the SemanticSearch class,
testing query expansion, search functionality, and related search suggestions.
"""

import pytest
from unittest.mock import patch, MagicMock, call
from langchain.schema import AIMessage

from src.search import SemanticSearch


class TestSemanticSearch:
    """Test cases for the SemanticSearch class."""
    
    @pytest.fixture
    def mock_db_service(self):
        """Create a mock DatabaseService."""
        with patch('src.search.DatabaseService') as mock_db_service:
            db_service_instance = MagicMock()
            mock_db_service.return_value = db_service_instance
            yield db_service_instance
    
    @pytest.fixture
    def mock_chat_model(self):
        """Create a mock ChatOpenAI model."""
        with patch('src.search.ChatOpenAI') as mock_chat:
            chat_instance = MagicMock()
            mock_chat.return_value = chat_instance
            yield chat_instance
    
    @pytest.fixture
    def search_service(self, mock_db_service, mock_chat_model):
        """Create a SearchService with mocked dependencies."""
        with patch('src.search.OpenAIConfig'):
            return SemanticSearch()
    
    def test_search_without_expansion(self, search_service, mock_db_service):
        """Test search functionality without query expansion."""
        # Arrange
        mock_db_service.search_articles.return_value = [
            {
                'id': '123',
                'title': 'Test Article',
                'summary': 'This is a test summary',
                'topics': ['technology', 'news'],
                'keywords': ['test', 'news'],
                'relevance_score': 0.85
            }
        ]
        
        # Act
        results = search_service.search("test query", expand_query=False)
        
        # Assert
        assert len(results) == 1
        assert results[0]['title'] == 'Test Article'
        assert results[0].get('relevance_percentage') == 85
        mock_db_service.search_articles.assert_called_once_with("test query", 5)
    
    def test_search_with_expansion(self, search_service, mock_db_service, mock_chat_model):
        """Test search functionality with query expansion."""
        # Arrange
        mock_chat_model.invoke.return_value = AIMessage(content="expanded test query with additional terms")
        mock_db_service.search_articles.return_value = [
            {
                'id': '123',
                'title': 'Test Article',
                'summary': 'This is a test summary',
                'topics': ['technology', 'news'],
                'keywords': 'test, news',  # Test string format
                'relevance_score': 0.85
            }
        ]
        
        # Act
        results = search_service.search("test query", expand_query=True)
        
        # Assert
        assert len(results) == 1
        assert results[0]['title'] == 'Test Article'
        assert isinstance(results[0]['keywords'], list)
        assert len(results[0]['keywords']) == 2
        mock_db_service.search_articles.assert_called_once_with("expanded test query with additional terms", 5)
    
    def test_get_related_searches(self, search_service, mock_chat_model):
        """Test related search suggestions."""
        # Arrange
        mock_chat_model.invoke.return_value = AIMessage(content="technology news\nlatest tech updates\nbreaking news")
        
        # Act
        suggestions = search_service.get_related_searches("tech news", num_suggestions=3)
        
        # Assert
        assert len(suggestions) == 3
        assert "technology news" in suggestions
        assert "latest tech updates" in suggestions
    
    def test_query_expansion_exception(self, search_service, mock_chat_model, mock_db_service):
        """Test handling of exceptions during query expansion."""
        # Arrange
        mock_chat_model.invoke.side_effect = Exception("API error")
        mock_db_service.search_articles.return_value = []
        
        # Act
        results = search_service.search("test query")
        
        # Assert
        mock_db_service.search_articles.assert_called_once_with("test query", 5)
        assert results == []
    
    def test_enhance_results(self, search_service):
        """Test enhancement of search results."""
        # Arrange
        results = [
            {
                'id': '123',
                'title': 'Test Article',
                'relevance_score': 0.75
            }
        ]
        
        # Act
        enhanced_results = search_service._enhance_results("query", results)
        
        # Assert
        assert enhanced_results[0]['relevance_percentage'] == 75
