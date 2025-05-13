"""
Streamlit UI module.

This module provides a web-based user interface for the news scraper application
using Streamlit, allowing users to input URLs, display article summaries and topics,
and perform semantic searches.
"""

import streamlit as st
import logging
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

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


def validate_url(url: str) -> bool:
    """
    Validate if a string is a properly formatted URL.
    
    Args:
        url: URL string to validate
        
    Returns:
        bool: True if valid URL, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def init_session_state():
    """Initialize session state variables if they don't exist."""
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    
    if "processed_urls" not in st.session_state:
        st.session_state.processed_urls = []
    
    if "related_searches" not in st.session_state:
        st.session_state.related_searches = []
        
    if "db_service" not in st.session_state:
        st.session_state.db_service = DatabaseService()
    
    if "search_service" not in st.session_state:
        st.session_state.search_service = SemanticSearch()


def display_article_card(article: Dict[str, Any]):
    """
    Display an article card in the UI.
    
    Args:
        article: Article data dictionary
    """
    with st.container():
        st.subheader(article.get('title', 'Untitled Article'))
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**Summary:**")
            st.write(article.get('summary', 'No summary available'))
            
            if 'url' in article:
                st.markdown(f"[Read original article]({article['url']})")
        
        with col2:
            st.markdown("**Topics:**")
            for topic in article.get('topics', []):
                st.markdown(f"- {topic}")
            
            if 'keywords' in article and article['keywords']:
                st.markdown("**Keywords:**")
                keywords_text = ", ".join(article['keywords'][:5])
                st.markdown(f"_{keywords_text}_")
            
            if 'relevance_percentage' in article:
                st.markdown(f"**Relevance:** {article['relevance_percentage']}%")
        
        st.divider()


def process_article_url(url: str) -> bool:
    """
    Process an article URL: extract content, generate summary, and store in database.
    
    Args:
        url: The URL of the article to process
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        with st.spinner("Processing article..."):
            # Initialize components
            extractor = ArticleExtractor()
            summarizer = ArticleSummarizer()
            
            # Extract article content
            article = extractor.extract(url)
            
            # Generate summary
            summary = summarizer.summarize(article)
            
            # Identify topics
            topics = summarizer.identify_topics(article)
            
            # Store article in database
            doc_id = st.session_state.db_service.store_article(article, summary, topics)
            
            # Add URL to processed list
            if url not in st.session_state.processed_urls:
                st.session_state.processed_urls.append(url)
            
            return True
    
    except Exception as e:
        st.error(f"Error processing article: {str(e)}")
        logger.error(f"Error processing article {url}: {str(e)}")
        return False


def display_all_articles():
    """Display all articles stored in the database."""
    try:
        # Get all articles
        articles = st.session_state.db_service.list_articles()
        
        if not articles:
            st.info("No articles found in the database. Add some articles first!")
            return
        
        st.subheader(f"All Articles ({len(articles)})")
        
        # Display each article
        for article in articles:
            display_article_card(article)
            
    except Exception as e:
        st.error(f"Error retrieving articles: {str(e)}")
        logger.error(f"Error retrieving articles: {str(e)}")


def search_articles(query: str, expand_query: bool = True):
    """
    Search for articles based on a query.
    
    Args:
        query: Search query string
        expand_query: Whether to use query expansion
    """
    try:
        with st.spinner("Searching..."):
            # Perform search
            results = st.session_state.search_service.search(query, expand_query=expand_query)
            st.session_state.search_results = results
            
            # Generate related searches
            related = st.session_state.search_service.get_related_searches(query)
            st.session_state.related_searches = related
    
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        logger.error(f"Search error: {str(e)}")


def main():
    """Main function to run the Streamlit application."""
    # Set page configuration
    st.set_page_config(
        page_title="News Scraper & Summarizer",
        page_icon="📰",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("📰 News Scraper & Summarizer")
    st.markdown("""
    This application extracts news articles from URLs, generates AI summaries, 
    identifies topics, and enables semantic search across stored articles.
    """)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Add Articles", "Browse Articles", "Search"])
    
    # Tab 1: Add Articles
    with tab1:
        st.header("Add News Articles")
        
        # URL input form
        with st.form("url_form"):
            url = st.text_input("Enter a news article URL")
            submitted = st.form_submit_button("Process Article")
            
            if submitted:
                if not validate_url(url):
                    st.error("Please enter a valid URL")
                else:
                    success = process_article_url(url)
                    if success:
                        st.success("Article processed and stored successfully!")
        
        # Recently processed URLs
        if st.session_state.processed_urls:
            st.subheader("Recently Processed URLs")
            for url in st.session_state.processed_urls[-5:]:
                st.markdown(f"- [{url}]({url})")
    
    # Tab 2: Browse Articles
    with tab2:
        st.header("Browse Articles")
        
        if st.button("Refresh Article List"):
            st.rerun()
        
        display_all_articles()
    
    # Tab 3: Search
    with tab3:
        st.header("Semantic Search")
        
        # Search form
        with st.form("search_form"):
            query = st.text_input("Search for articles")
            col1, col2 = st.columns(2)
            
            with col1:
                expand_query = st.checkbox("Use query expansion", value=True,
                                         help="Enhance search with AI-powered query expansion")
            
            submitted = st.form_submit_button("Search")
            
            if submitted and query:
                search_articles(query, expand_query)
        
        # Display search results
        if st.session_state.search_results:
            st.subheader(f"Search Results ({len(st.session_state.search_results)})")
            
            for article in st.session_state.search_results:
                display_article_card(article)
        
        # Display related searches
        if st.session_state.related_searches:
            st.subheader("Related Searches")
            for search in st.session_state.related_searches:
                if st.button(search, key=f"related_{hash(search)}"):
                    search_articles(search)


if __name__ == "__main__":
    main()
