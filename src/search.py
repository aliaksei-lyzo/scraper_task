"""
Semantic search module.

This module provides enhanced semantic search capabilities using GenAI 
and ChromaDB to find relevant articles based on user queries.
"""

import logging
from typing import List, Dict, Any
from pydantic import SecretStr

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from src.config import OpenAIConfig
from src.database import DatabaseService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemanticSearch:
    """
    Enhanced semantic search capabilities using GenAI and ChromaDB.
    
    This class provides methods to search for relevant articles based on user 
    queries, with additional features like query expansion and result ranking.
    """
    
    def __init__(self):
        """Initialize the SemanticSearch with database service and LLM."""
        # Validate OpenAI configuration
        OpenAIConfig.validate()
        
        # Initialize the database service
        self.db_service = DatabaseService()
        
        # Initialize the ChatOpenAI model
        self.model = ChatOpenAI(
            model=OpenAIConfig.MODEL,
            temperature=0.2,  # Lower temperature for more predictable expansions
            api_key=SecretStr(OpenAIConfig.API_KEY),
        )
        logger.info(f"Initialized SemanticSearch with model: {OpenAIConfig.MODEL}")
    
    def search(self, query: str, limit: int = 5, expand_query: bool = True) -> List[Dict[str, Any]]:
        """
        Search for articles based on semantic similarity to the query.
        
        Args:
            query: The search query text
            limit: Maximum number of results to return
            expand_query: Whether to use query expansion with GenAI
            
        Returns:
            List of search results with article data
            
        Raises:
            ValueError: If the search fails
        """
        try:
            if expand_query:
                # Expand the query using GenAI for better semantic matching
                expanded_query = self._expand_query(query)
                logger.info(f"Expanded query '{query}' to '{expanded_query}'")
                search_results = self.db_service.search_articles(expanded_query, limit)
            else:
                # Use the original query directly
                search_results = self.db_service.search_articles(query, limit)
            
            # Enhance the search results with additional info
            enhanced_results = self._enhance_results(query, search_results)
            
            logger.info(f"Search for '{query}' returned {len(enhanced_results)} results")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error performing semantic search: {str(e)}")
            raise ValueError(f"Failed to search for articles: {str(e)}")
    
    def _expand_query(self, query: str) -> str:
        """
        Expand the query using GenAI for better semantic matching.
        
        Args:
            query: The original search query
            
        Returns:
            Expanded query string
        """
        try:
            # Create messages for the chat model
            messages = [
                SystemMessage(content=(
                    "You are a helpful assistant that expands search queries to improve semantic search results. "
                    "Given a search query, expand it with relevant terms and synonyms while preserving the "
                    "original intent. Keep the expansion concise and focused."
                )),
                HumanMessage(content=(
                    f"Original query: {query}\n"
                    "Please expand this query for better semantic search results. "
                    "Return ONLY the expanded query text, no explanations or formatting."
                ))
            ]
            
            # Get response from model
            response = self.model.invoke(messages)
            content = response.content
            if isinstance(content, str):
                expanded_query = content.strip()
            else:
                expanded_query = " ".join(item.strip() for item in content if isinstance(item, str))

            # Avoid overly long expansions
            if len(expanded_query) > 200:
                expanded_query = expanded_query[:200]
            
            return expanded_query
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {str(e)}")
            return query  # Fall back to the original query
    
    def _enhance_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance search results with additional information.
        
        Args:
            query: The original search query
            results: The initial search results
            
        Returns:
            Enhanced search results
        """
        enhanced_results = []
        
        for result in results:
            # Calculate relevance percentage from score (if available)
            if result.get('relevance_score') is not None:
                result['relevance_percentage'] = int(result['relevance_score'] * 100)
            
            # Ensure keywords are properly split
            if isinstance(result.get('keywords'), str):
                result['keywords'] = result['keywords'].split(", ")
            
            enhanced_results.append(result)
        
        return enhanced_results
    
    def get_related_searches(self, query: str, num_suggestions: int = 3) -> List[str]:
        """
        Generate related search suggestions based on the original query.
        
        Args:
            query: The original search query
            num_suggestions: Number of related search suggestions to generate
            
        Returns:
            List of related search suggestions
            
        Raises:
            ValueError: If generation fails
        """
        try:
            # Create messages for the chat model
            messages = [
                SystemMessage(content=(
                    "You are a helpful search assistant that suggests related searches. "
                    f"Given a search query, provide {num_suggestions} alternative search queries "
                    "that a user might be interested in. Each suggestion should be on a new line."
                )),
                HumanMessage(content=(
                    f"Original search query: {query}\n"
                    f"Generate {num_suggestions} related search suggestions, one per line."
                ))
            ]
            
            # Get response from model
            response = self.model.invoke(messages)
            content = response.content
            
            # Handle content which could be string or list
            if isinstance(content, str):
                suggestions_text = content.strip()
            else:
                suggestions_text = " ".join(item.strip() for item in content if isinstance(item, str))
            
            # Parse the response into a list of suggestions
            suggestions = [
                line.strip().strip('-*• \t') 
                for line in suggestions_text.split('\n') 
                if line.strip()
            ]
            
            # Limit to the requested number and ensure unique suggestions
            unique_suggestions = list(dict.fromkeys(suggestions))[:num_suggestions]
            
            logger.info(f"Generated {len(unique_suggestions)} related search suggestions for '{query}'")
            return unique_suggestions
            
        except Exception as e:
            logger.error(f"Error generating related searches: {str(e)}")
            raise ValueError(f"Failed to generate related searches: {str(e)}")
