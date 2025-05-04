"""
News article extractor module.

This module provides functionality to extract headlines and full text content
from news article URLs using requests and BeautifulSoup.
"""

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArticleContent(BaseModel):
    """
    Pydantic model for article content extracted from a URL.
    
    Attributes:
        url: The source URL of the article
        title: The headline/title of the article
        text: The full text content of the article
        metadata: Additional metadata like author, publication date, etc.
    """
    url: HttpUrl
    title: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


class ArticleExtractor:
    """
    Extracts news article content from URLs.
    
    This class provides methods to fetch and parse HTML content from news websites
    to extract article headlines and full text.
    """
    
    def __init__(self):
        """Initialize the ArticleExtractor with default headers."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                         '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract(self, url: str) -> ArticleContent:
        """
        Extract headline and full text from a news article URL.
        
        Args:
            url: The URL of the news article to extract content from
            
        Returns:
            ArticleContent: Contains the article title, text, and metadata
            
        Raises:
            ValueError: If the URL cannot be fetched or content cannot be extracted
        """
        try:
            logger.info(f"Extracting content from: {url}")
            
            # Fetch the HTML content
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title - common patterns in news sites
            title = self._extract_title(soup)
            
            # Extract content - common patterns in news sites
            text = self._extract_text(soup)
            
            # Extract metadata (could be expanded)
            metadata = self._extract_metadata(soup)
            
            return ArticleContent(
                url=url,
                title=title,
                text=text,
                metadata=metadata
            )
        
        except requests.RequestException as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            raise ValueError(f"Failed to fetch article from {url}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            raise ValueError(f"Failed to extract article content from {url}: {str(e)}")
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """
        Extract the article title from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object of the article page
            
        Returns:
            str: The extracted title
            
        Raises:
            ValueError: If title cannot be found
        """
        # Try different common patterns for article titles
        # 1. Look for article tags with headlines
        if article := soup.find('article'):
            if headline := article.find(['h1', 'h2']):
                return headline.get_text().strip()
        
        # 2. Look for standard header tags
        if h1 := soup.find('h1'):
            return h1.get_text().strip()
        
        # 3. Look for common title classes/IDs
        for selector in ['.headline', '.article-title', '#article-title', '.post-title', '.entry-title']:
            if title_elem := soup.select_one(selector):
                return title_elem.get_text().strip()
        
        # 4. Fall back to HTML title tag
        if html_title := soup.title:
            return html_title.get_text().strip()
        
        # If we get here, no title was found
        raise ValueError("Could not extract article title")
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extract the full text content from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object of the article page
            
        Returns:
            str: The extracted article text
            
        Raises:
            ValueError: If article text cannot be found
        """
        content = ""
        
        # 1. Look for article content in main content areas
        content_selectors = [
            'article', '.article-body', '.story-body', '.post-content', 
            '.entry-content', '#article-body', '.story-content', 'main'
        ]
        
        for selector in content_selectors:
            if content_area := soup.select_one(selector):
                # Get all paragraphs in the content area
                paragraphs = content_area.find_all('p')
                if paragraphs:
                    content = '\n\n'.join([p.get_text().strip() for p in paragraphs])
                    break
        
        # 2. Fall back to all paragraphs if needed
        if not content:
            # Exclude navigation, footer, etc.
            exclude_selectors = ['nav', 'footer', 'header', '.comments', '.sidebar']
            for selector in exclude_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Get remaining paragraphs
            paragraphs = soup.find_all('p')
            content = '\n\n'.join([p.get_text().strip() for p in paragraphs])
        
        if not content:
            raise ValueError("Could not extract article text")
        
        return content
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract metadata from the article.
        
        Args:
            soup: BeautifulSoup object of the article page
            
        Returns:
            Dict[str, Any]: Dictionary of metadata fields
        """
        metadata = {}
        
        # Try to extract author
        author_selectors = [
            '.author', '.byline', '.article-author', 
            'meta[name="author"]', 'meta[property="article:author"]'
        ]
        
        for selector in author_selectors:
            if 'meta' in selector:
                if author_meta := soup.select_one(selector):
                    metadata['author'] = author_meta.get('content', '').strip()
                    break
            elif author_elem := soup.select_one(selector):
                metadata['author'] = author_elem.get_text().strip()
                break
        
        # Try to extract date
        date_selectors = [
            '.date', '.published-date', '.article-date', '.timestamp',
            'time', 'meta[property="article:published_time"]'
        ]
        
        for selector in date_selectors:
            if 'meta' in selector:
                if date_meta := soup.select_one(selector):
                    metadata['date'] = date_meta.get('content', '').strip()
                    break
            elif date_elem := soup.select_one(selector):
                # Check if there's a datetime attribute
                if date_attr := date_elem.get('datetime'):
                    metadata['date'] = date_attr.strip()
                else:
                    metadata['date'] = date_elem.get_text().strip()
                break
        
        return metadata
