# News Scraper with GenAI

A Python application that extracts news articles from URLs, generates summaries and identifies topics using GenAI tools, and enables semantic search with ChromaDB.

## Features

- Extract headlines and full text from news article URLs
- Generate concise or detailed summaries using LangChain and OpenAI
- Identify main topics and keywords from articles
- Store articles, summaries, and topics in ChromaDB vector database
- Perform semantic search based on natural language queries
- Simple command-line interface for demonstration

## Architecture

The application consists of the following components:

- **Article Extraction**: Extracts news article content from URLs using BeautifulSoup.
- **GenAI Summarization**: Uses LangChain and OpenAI to generate article summaries.
- **Topic Identification**: Uses LangChain and OpenAI to identify main topics and keywords.
- **Vector Database**: Stores articles with embeddings in ChromaDB for semantic search.

## Setup

### Prerequisites

- Python 3.8 or higher
- Docker (for ChromaDB)
- Poetry (for package management)
- OpenAI API key

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/news-scraper.git
cd news-scraper
```

2. Install dependencies with Poetry:

```bash
poetry install
```

3. Create a `.env` file in the root directory with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=500
```

4. Start the ChromaDB server using Docker:

```bash
docker-compose up -d
```

### Usage

To run the application with example articles:

```bash
poetry run python -m src.main
```

This will:
1. Extract content from example news articles
2. Generate summaries and identify topics
3. Store the articles in ChromaDB
4. Demonstrate semantic search capabilities

## Project Structure

```
news-scraper/
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration and environment variables
│   ├── extractor.py      # News article extraction logic
│   ├── summarizer.py     # Article summarization with LangChain
│   ├── database.py       # ChromaDB integration for vector database
│   ├── models.py         # Pydantic data models
│   └── main.py           # Main application entry point
├── tests/                # Unit tests
├── data/                 # Data storage (ChromaDB files)
├── docker-compose.yml    # Docker setup for ChromaDB
├── .env                  # Environment variables (not in repo)
├── pyproject.toml        # Poetry dependency management
└── README.md             # This file
```

## Testing

To run the tests:

```bash
poetry run pytest
```

## License

[MIT License](LICENSE)