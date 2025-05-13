"""
Database service module for ChromaDB interactions.

This module provides functionality to interact with ChromaDB for storing
and retrieving article data with vector embeddings for semantic search.
"""

import logging
import uuid
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import SecretStr

import chromadb
from langchain_openai import OpenAIEmbeddings

from src.config import OpenAIConfig
from src.models import ArticleContent, ArticleSummary, TopicIdentification, ArticleDocument

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Service for interacting with ChromaDB vector database.
    
    This class provides methods to store and retrieve article data with 
    vector embeddings for semantic search capabilities.
    """
    
    # Collection names constants
    ARTICLES_COLLECTION = "articles"
    def __init__(self):
        """
        Initialize the DatabaseService.
        """        
        persist_directory = os.path.join(Path(__file__).parents[1], "data", "chroma")
            
        os.makedirs(persist_directory, exist_ok=True)
        
        logger.info(f"Initializing local ChromaDB client with persistence at {persist_directory}")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = OpenAIEmbeddings(
            api_key=SecretStr(OpenAIConfig.API_KEY),
            model="text-embedding-ada-002"
        )
        
        self._init_collections()
        
        logger.info("DatabaseService initialized successfully")
        
    def _init_collections(self) -> None:
        """
        Initialize the default collections in ChromaDB.
        
        This creates the articles collection if it doesn't exist.
        """
        try:
            try:
                self.client.get_collection(name=self.ARTICLES_COLLECTION)
                logger.info(f"Collection '{self.ARTICLES_COLLECTION}' already exists")
            except ValueError:
                self.client.create_collection(
                    name=self.ARTICLES_COLLECTION,
                    metadata={"description": "Collection for news articles with summaries and topics"}
                )
                logger.info(f"Created collection '{self.ARTICLES_COLLECTION}'")
        except Exception as e:
            logger.error(f"Error initializing collections: {str(e)}")
            raise
    
    def store_article(
        self, 
        article: ArticleContent, 
        summary: ArticleSummary, 
        topics: TopicIdentification
    ) -> str:
        """
        Store an article with its summary and topics in ChromaDB.
        
        Args:
            article: The article content
            summary: The article summary
            topics: The article topics and keywords
            
        Returns:
            str: The ID of the stored document
            
        Raises:
            ValueError: If storing fails
        """
        try:
            # Generate a unique ID for the article
            doc_id = str(uuid.uuid4())
            
            # Create document for storage
            document = ArticleDocument(
                id=doc_id,
                content=article,
                summary=summary.summary,
                topics=topics.topics,
                keywords=topics.keywords
            )
            
            # Get the collection
            collection = self.client.get_collection(name=self.ARTICLES_COLLECTION)
            
            # Prepare text for embedding - combine important elements for better semantic search
            text_for_embedding = (
                f"Title: {article.title}\n"
                f"Summary: {summary.summary}\n"
                f"Topics: {', '.join(topics.topics)}\n"
                f"Keywords: {', '.join(topics.keywords)}"
            )
            
            # Get embedding directly from OpenAI
            embedding = self.embedding_function.embed_query(text_for_embedding)
            
            # Convert article data to strings for storage
            document_data = {
                "url": str(article.url),
                "title": article.title,
                "text": article.text[:1000],  # Store truncated text to avoid size limitations
                "summary": summary.summary,
                "topics": ", ".join(topics.topics),
                "keywords": ", ".join(topics.keywords)
            }
            
            # Store the document
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[document_data],
                documents=[text_for_embedding]
            )
            
            logger.info(f"Successfully stored article '{article.title}' with ID {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error storing article: {str(e)}")
            raise ValueError(f"Failed to store article: {str(e)}")
    
    def search_articles(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for articles based on semantic similarity to the query.
        
        Args:
            query: The search query text
            limit: Maximum number of results to return
            
        Returns:
            List of search results with article data
            
        Raises:
            ValueError: If the search fails
        """
        try:
            # Get the collection
            collection = self.client.get_collection(name=self.ARTICLES_COLLECTION)
            
            # Get embedding for query from OpenAI
            query_embedding = self.embedding_function.embed_query(query)
            
            # Search the collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
            
            # Process and format results
            formatted_results = []
            
            if results["ids"] and len(results["ids"][0]) > 0:
                for i, doc_id in enumerate(results["ids"][0]):
                    # Extract data from the results
                    metadata = results["metadatas"][0][i] if results["metadatas"] and len(results["metadatas"][0]) > i else {}
                    distance = results["distances"][0][i] if results["distances"] and len(results["distances"][0]) > i else None
                    
                    # Format the result
                    formatted_result = {
                        "id": doc_id,
                        "url": metadata.get("url", ""),
                        "title": metadata.get("title", ""),
                        "summary": metadata.get("summary", ""),
                        "topics": str(metadata.get("topics", "")).split(", "),
                        "keywords": str(metadata.get("keywords", "")).split(", "),
                        "relevance_score": 1 - (distance or 0) if distance is not None else None
                    }
                    
                    formatted_results.append(formatted_result)
            
            logger.info(f"Search for '{query}' returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching for articles: {str(e)}")
            raise ValueError(f"Failed to search for articles: {str(e)}")
    
    def get_article_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an article by its document ID.
        
        Args:
            doc_id: The ID of the document to retrieve
            
        Returns:
            Dict with article data or None if not found
            
        Raises:
            ValueError: If retrieval fails
        """
        try:
            # Get the collection
            collection = self.client.get_collection(name=self.ARTICLES_COLLECTION)
            
            # Get the document by ID
            result = collection.get(ids=[doc_id])
            
            # Check if document was found
            if not result["ids"]:
                logger.warning(f"Article with ID {doc_id} not found")
                return None
            
            # Extract data from the result
            metadata = result["metadatas"][0] if result["metadatas"] else {}
            
            # Format the result
            article_data = {
                "id": doc_id,
                "url": metadata.get("url", ""),
                "title": metadata.get("title", ""),
                "summary": metadata.get("summary", ""),
                "topics": str(metadata.get("topics", "")).split(", "),
                "keywords": str(metadata.get("keywords", "")).split(", ")
            }
            
            logger.info(f"Successfully retrieved article with ID {doc_id}")
            return article_data
            
        except Exception as e:
            logger.error(f"Error retrieving article with ID {doc_id}: {str(e)}")
            raise ValueError(f"Failed to retrieve article: {str(e)}")
    
    def list_articles(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all stored articles.
        
        Args:
            limit: Maximum number of articles to return
            
        Returns:
            List of articles with basic data
            
        Raises:
            ValueError: If listing fails
        """
        try:
            # Get the collection
            collection = self.client.get_collection(name=self.ARTICLES_COLLECTION)
            
            # Get all documents (up to limit)
            result = collection.get(limit=limit)
            
            # Format the results
            articles = []
            
            for i, doc_id in enumerate(result["ids"]):
                # Extract data from the result
                metadata = result["metadatas"][i] if result["metadatas"] else {}
                
                # Format the result
                article_data = {
                    "id": doc_id,
                    "url": metadata.get("url", ""),
                    "title": metadata.get("title", ""),
                    "topics": str(metadata.get("topics", "")).split(", ")
                }
                
                articles.append(article_data)
            
            logger.info(f"Listed {len(articles)} articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error listing articles: {str(e)}")
            raise ValueError(f"Failed to list articles: {str(e)}")
    
    def delete_article(self, doc_id: str) -> bool:
        """
        Delete an article by its document ID.
        
        Args:
            doc_id: The ID of the document to delete
            
        Returns:
            bool: True if deletion was successful
            
        Raises:
            ValueError: If deletion fails
        """
        try:
            # Get the collection
            collection = self.client.get_collection(name=self.ARTICLES_COLLECTION)
            
            # Delete the document
            collection.delete(ids=[doc_id])
            
            logger.info(f"Successfully deleted article with ID {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting article with ID {doc_id}: {str(e)}")
            raise ValueError(f"Failed to delete article: {str(e)}")
    
    def reset_database(self) -> bool:
        """
        Reset the entire database (delete all collections).
        
        Returns:
            bool: True if reset was successful
            
        Raises:
            ValueError: If reset fails
        """
        try:
            # Reset the client (delete all collections)
            self.client.reset()
            
            # Re-initialize collections
            self._init_collections()
            
            logger.info("Successfully reset the database")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting database: {str(e)}")
            raise ValueError(f"Failed to reset database: {str(e)}")
