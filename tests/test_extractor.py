"""
Tests for the article extractor module.
"""

import pytest
from unittest.mock import patch, Mock
import requests
from bs4 import BeautifulSoup
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.extractor import ArticleExtractor
from src.models import ArticleContent


@pytest.fixture
def mock_response():
    """Create a mock HTTP response with sample HTML."""
    mock_resp = Mock()
    mock_resp.text = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Article Title</title>
        <meta name="author" content="John Doe">
        <meta property="article:published_time" content="2025-05-04T12:00:00Z">
    </head>
    <body>
        <article>
            <h1 class="headline">Sample News Article</h1>
            <div class="article-body">
                <p>This is the first paragraph of the article.</p>
                <p>This is the second paragraph with more details.</p>
                <p>This is the conclusion of the article.</p>
            </div>
        </article>
        <footer>
            <p>Copyright 2025</p>
        </footer>
    </body>
    </html>
    """
    mock_resp.status_code = 200
    return mock_resp


def test_extract_success(mock_response):
    """Test successful article extraction."""
    with patch('requests.get', return_value=mock_response):
        extractor = ArticleExtractor()
        result = extractor.extract('https://example.com/article')
        
        assert isinstance(result, ArticleContent)
        assert result.title == "Sample News Article"
        assert "first paragraph" in result.text
        assert "second paragraph" in result.text
        assert "conclusion" in result.text
        assert "Copyright" not in result.text  # Footer content should be excluded
        assert result.metadata.get('author') == "John Doe"
        assert result.metadata.get('date') == "2025-05-04T12:00:00Z"


def test_extract_request_error():
    """Test error handling when request fails."""
    with patch('requests.get', side_effect=requests.RequestException("Connection error")):
        extractor = ArticleExtractor()
        with pytest.raises(ValueError) as excinfo:
            extractor.extract('https://example.com/article')
        assert "Failed to fetch article" in str(excinfo.value)


def test_extract_no_title():
    """Test error handling when no title can be found."""
    mock_resp = Mock()
    mock_resp.text = "<html><body><p>Just some text</p></body></html>"
    mock_resp.status_code = 200
    
    with patch('requests.get', return_value=mock_resp):
        extractor = ArticleExtractor()
        with pytest.raises(ValueError) as excinfo:
            extractor.extract('https://example.com/article')
        assert "Could not extract article title" in str(excinfo.value)


def test_extract_no_content():
    """Test error handling when no content can be found."""
    mock_resp = Mock()
    mock_resp.text = "<html><head><title>Test</title></head><body></body></html>"
    mock_resp.status_code = 200
    
    with patch('requests.get', return_value=mock_resp):
        extractor = ArticleExtractor()
        with pytest.raises(ValueError) as excinfo:
            extractor.extract('https://example.com/article')
        assert "Could not extract article text" in str(excinfo.value)
