"""
Main application module for news article scraping.

This module demonstrates how to use the ArticleExtractor class
to extract content from news articles and the ArticleSummarizer
to generate summaries and identify topics.
"""

import logging
from extractor import ArticleExtractor, ArticleContent
from summarizer import ArticleSummarizer, ArticleSummary, TopicIdentification


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main function to demonstrate article extraction, summarization, and topic identification.
    
    Extracts, summarizes, and analyzes a sample BBC news article.
    """
    # Initialize the article extractor and summarizer
    extractor = ArticleExtractor()
    summarizer = ArticleSummarizer()
    
    # BBC News article URL
    url = "https://www.bbc.com/news/articles/c3r807j7xrwo"
    
    try:
        # Extract article content
        logger.info(f"Attempting to extract content from: {url}")
        article: ArticleContent = extractor.extract(url)
        
        # Display the extracted information
        print("\n===== EXTRACTED ARTICLE =====")
        print(f"Title: {article.title}")
        print("\n--- Full Text ---")
        print(f"{article.text[:500]}...")  # Only show first 500 chars
        print("\n--- Metadata ---")
        if article.metadata:
            for key, value in article.metadata.items():
                print(f"{key}: {value}")
        else:
            print("No metadata extracted")
        
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
        
        logger.info("Article processing completed successfully")
            
    except ValueError as e:
        logger.error(f"Extraction failed: {e}")
        print(f"Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()