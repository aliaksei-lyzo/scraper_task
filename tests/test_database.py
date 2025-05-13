"""
Tests for the database service module.

This module contains tests for the DatabaseService class that handles
interactions with the ChromaDB vector database.
"""

import unittest
from unittest.mock import patch, MagicMock
import uuid
import os
from pydantic import HttpUrl

from src.models import ArticleContent, ArticleSummary, TopicIdentification
from src.database import DatabaseService


class TestDatabaseService(unittest.TestCase):
    """Test cases for the DatabaseService class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""        # Create mock article data for testing
        self.article = ArticleContent(
            url=HttpUrl("https://www.example.com/article"),
            title="Test Article Title",
            text="This is a test article with enough content to process.",
            metadata={"author": "Test Author"}
        )
        article_id = str(hash(f"{self.article.url}-{self.article.title}"))
        
        self.summary = ArticleSummary(
            article_id=article_id,
            summary="This is a summary of the test article.",
            summary_type="concise"
        )
        
        self.topics = TopicIdentification(
            article_id=article_id,
            topics=["Technology", "Testing", "AI"],
            keywords=["test", "article", "mock", "database", "vector"]
        )
        
        # Mock UUID generation to return consistent IDs for testing
        self.mock_uuid = "test-uuid-12345"
        uuid.uuid4 = MagicMock(return_value=self.mock_uuid)
        
    @patch('chromadb.PersistentClient')
    @patch('langchain_openai.OpenAIEmbeddings')
    def test_init_client(self, mock_embeddings, mock_client):
        """Test DatabaseService initialization with local persistent client."""
        # Setup mocks
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collection.side_effect = ValueError()  # Simulate collection doesn't exist
        
        # Initialize service
        service = DatabaseService()
        
        # Assert that the client was created correctly
        mock_client.assert_called_once()
        mock_client_instance.create_collection.assert_called_once_with(
            name=DatabaseService.ARTICLES_COLLECTION,
            metadata={"description": "Collection for news articles with summaries and topics"}
        )
        
    @patch('chromadb.PersistentClient')
    @patch('langchain_openai.OpenAIEmbeddings')
    def test_store_article(self, mock_embeddings, mock_client):
        """Test storing an article in the database."""
        # Setup mocks
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_collection = MagicMock()
        mock_client_instance.get_collection.return_value = mock_collection
        
        # Mock embeddings
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance
        mock_embedding_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Initialize service and store article
        service = DatabaseService()
        result = service.store_article(self.article, self.summary, self.topics)
        
        # Assert correct storage
        self.assertEqual(result, str(self.mock_uuid))
        mock_embedding_instance.embed_query.assert_called_once()
        mock_collection.add.assert_called_once()
        self.assertEqual(mock_collection.add.call_args[1]['ids'], [str(self.mock_uuid)])
        
    @patch('chromadb.PersistentClient')
    @patch('langchain_openai.OpenAIEmbeddings')
    def test_search_articles(self, mock_embeddings, mock_client):
        """Test searching for articles."""
        # Setup mocks
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_collection = MagicMock()
        mock_client_instance.get_collection.return_value = mock_collection
        
        # Mock query results
        mock_collection.query.return_value = {
            "ids": [["doc-id-1", "doc-id-2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[
                {
                    "url": "https://example.com/1",
                    "title": "Article 1",
                    "summary": "Summary 1",
                    "topics": "Tech, AI",
                    "keywords": "ai, tech, test"
                },
                {
                    "url": "https://example.com/2",
                    "title": "Article 2",
                    "summary": "Summary 2",
                    "topics": "News, Politics",
                    "keywords": "news, politics, world"
                }
            ]]
        }
        
        # Mock embeddings
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance
        mock_embedding_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Initialize service and search articles
        service = DatabaseService()
        results = service.search_articles("test search query")
        
        # Assert correct search results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "doc-id-1")
        self.assertEqual(results[0]["title"], "Article 1")
        self.assertEqual(results[0]["topics"], ["Tech", "AI"])
        self.assertEqual(results[1]["id"], "doc-id-2")
        self.assertEqual(results[1]["title"], "Article 2")
        
    @patch('chromadb.PersistentClient')
    @patch('langchain_openai.OpenAIEmbeddings')
    def test_get_article_by_id(self, mock_embeddings, mock_client):
        """Test retrieving an article by ID."""
        # Setup mocks
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_collection = MagicMock()
        mock_client_instance.get_collection.return_value = mock_collection
        
        # Mock get results
        mock_collection.get.return_value = {
            "ids": ["doc-id-1"],
            "metadatas": [
                {
                    "url": "https://example.com/1",
                    "title": "Article 1",
                    "summary": "Summary 1",
                    "topics": "Tech, AI",
                    "keywords": "ai, tech, test"
                }
            ]
        }
        
        # Initialize service and get article
        service = DatabaseService()
        result = service.get_article_by_id("doc-id-1") or {}
        
        # Assert correct article retrieval
        self.assertEqual(result["id"], "doc-id-1")
        self.assertEqual(result["title"], "Article 1")
        self.assertEqual(result["topics"], ["Tech", "AI"])
        
    @patch('chromadb.PersistentClient')
    @patch('langchain_openai.OpenAIEmbeddings')
    def test_get_article_by_id_not_found(self, mock_embeddings, mock_client):
        """Test retrieving a non-existent article by ID."""
        # Setup mocks
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_collection = MagicMock()
        mock_client_instance.get_collection.return_value = mock_collection
        
        # Mock get results for non-existent article
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        
        # Initialize service and get non-existent article
        service = DatabaseService()
        result = service.get_article_by_id("non-existent-id")
        
        # Assert correct behavior for non-existent article
        self.assertIsNone(result)
        
    @patch('chromadb.PersistentClient')
    @patch('langchain_openai.OpenAIEmbeddings')
    def test_delete_article(self, mock_embeddings, mock_client):
        """Test deleting an article."""
        # Setup mocks
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_collection = MagicMock()
        mock_client_instance.get_collection.return_value = mock_collection
        
        # Initialize service and delete article
        service = DatabaseService()
        result = service.delete_article("doc-id-1")
        
        # Assert correct deletion
        self.assertTrue(result)
        mock_collection.delete.assert_called_once_with(ids=["doc-id-1"])
        
    @patch('chromadb.PersistentClient')
    @patch('langchain_openai.OpenAIEmbeddings')
    def test_reset_database(self, mock_embeddings, mock_client):
        """Test resetting the database."""
        # Setup mocks
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Initialize service and reset database
        service = DatabaseService()
        result = service.reset_database()
        
        # Assert correct reset
        self.assertTrue(result)
        mock_client_instance.reset.assert_called_once()
        mock_client_instance.create_collection.assert_called_once()


if __name__ == '__main__':
    unittest.main()
