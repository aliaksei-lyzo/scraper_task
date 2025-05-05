import logging
from extractor import ArticleExtractor, ArticleContent

"""
Main application module for news article scraping.

This module demonstrates how to use the ArticleExtractor class
to extract content from news articles.
"""


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main function to demonstrate article extraction.
    
    Extracts and displays content from a sample BBC news article.
    """
    # Initialize the article extractor
    extractor = ArticleExtractor()
    
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
        print(article.text)
        print("\n--- Metadata ---")
        if article.metadata:
            for key, value in article.metadata.items():
                print(f"{key}: {value}")
        else:
            print("No metadata extracted")
        
        logger.info("Article extraction completed successfully")
            
    except ValueError as e:
        logger.error(f"Extraction failed: {e}")
        print(f"Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()