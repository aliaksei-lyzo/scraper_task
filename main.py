"""
Main application module for news article scraping.

This module demonstrates how to use the ArticleExtractor class
to extract content from news articles, the ArticleSummarizer
to generate summaries and identify topics, the DatabaseService
to store and retrieve articles from ChromaDB, and the SemanticSearch
service to perform enhanced semantic searches.
"""

import logging
import time
from src.extractor import ArticleExtractor
from src.summarizer import ArticleSummarizer
from src.database import DatabaseService
from src.search import SemanticSearch
from src.models import ArticleContent, ArticleSummary, TopicIdentification


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_article(url: str) -> tuple:
    """
    Process an article by extracting, summarizing, and identifying topics.
    
    Args:
        url: URL of the article to process
        
    Returns:
        tuple: (ArticleContent, ArticleSummary, TopicIdentification)
        
    Raises:
        ValueError: If processing fails
    """
    # Initialize the article extractor and summarizer
    extractor = ArticleExtractor()
    summarizer = ArticleSummarizer()
    
    # Extract article content
    logger.info(f"Attempting to extract content from: {url}")
    article: ArticleContent = extractor.extract(url)
    
    # Display the extracted information
    print("\n===== EXTRACTED ARTICLE =====")
    print(f"Title: {article.title}")
    print("\n--- Full Text ---")
    print(f"{article.text[:500]}...")  # Only show first 500 chars
    
    # Generate a concise summary
    logger.info("Generating article summary...")
    summary: ArticleSummary = summarizer.summarize(article, summary_type="concise")
    
    # Display the summary
    print("\n===== ARTICLE SUMMARY =====")
    print(summary.summary)
    
    # Identify topics and keywords
    logger.info("Identifying article topics...")
    topics: TopicIdentification = summarizer.identify_topics(article)
    
    # Display topics and keywords
    print("\n===== ARTICLE TOPICS =====")
    print(f"Topics: {', '.join(topics.topics)}")
    print(f"Keywords: {', '.join(topics.keywords)}")
    
    return article, summary, topics


def main() -> None:
    """
    Main function to demonstrate article extraction, summarization, topic identification,
    and vector database storage with semantic search.
    
    Processes sample news articles and demonstrates ChromaDB storage and search capabilities.
    """
    try:        # Initialize the database service
        logger.info("Initializing database service...")
        db_service = DatabaseService()
        
        # Process BBC News article
        url1 = "https://www.bbc.com/news/articles/c3r807j7xrwo"
        print(f"\nProcessing article: {url1}")
        article1, summary1, topics1 = process_article(url1)
        
        # Store the article in ChromaDB
        logger.info("Storing article in ChromaDB...")
        doc_id1 = db_service.store_article(article1, summary1, topics1)
        print(f"\n✅ Article stored in ChromaDB with ID: {doc_id1}")
        
        # Process a second article (for demo purposes)
        url2 = "https://www.bbc.com/news/technology-67046858"
        print(f"\nProcessing article: {url2}")
        article2, summary2, topics2 = process_article(url2)
        
        # Store the second article in ChromaDB
        logger.info("Storing second article in ChromaDB...")
        doc_id2 = db_service.store_article(article2, summary2, topics2)
        print(f"\n✅ Article stored in ChromaDB with ID: {doc_id2}")
        
        # Give time for embeddings to be processed
        print("\nWaiting for embeddings to be processed...")
        time.sleep(2)
          # Demonstrate semantic search capabilities
        print("\n===== SEMANTIC SEARCH DEMO =====")
        search_query = "Latest technology news"
        print(f"Searching for: '{search_query}'")
        
        # Initialize the semantic search service
        search_service = SemanticSearch()
        
        # Search with query expansion
        print("\n--- Enhanced Semantic Search with Query Expansion ---")
        search_results = search_service.search(search_query, expand_query=True)
        print(f"\nFound {len(search_results)} results:")
        
        for i, result in enumerate(search_results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Title: {result['title']}")
            if result.get('relevance_percentage') is not None:
                print(f"Relevance: {result['relevance_percentage']}%")
            else:
                print(f"Relevance Score: {result.get('relevance_score', 'N/A'):.2f}")
            print(f"Summary: {result['summary'][:150]}...")
            print(f"Topics: {', '.join(result['topics'])}")
            
        # Get related search suggestions
        related_searches = search_service.get_related_searches(search_query)
        print("\n--- Related Search Suggestions ---")
        for i, suggestion in enumerate(related_searches, 1):
            print(f"{i}. {suggestion}")
        
        # List all articles in the database
        print("\n===== ALL STORED ARTICLES =====")
        all_articles = db_service.list_articles()
        print(f"Total articles in database: {len(all_articles)}")
        
        for i, article in enumerate(all_articles, 1):
            print(f"\n{i}. {article['title']} (ID: {article['id']})")
            print(f"   Topics: {', '.join(article['topics'])}")
        
        logger.info("Demo completed successfully")
            
    except ValueError as e:
        logger.error(f"Processing failed: {e}")
        print(f"Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()