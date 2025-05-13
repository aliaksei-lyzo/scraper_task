# News Scraper with GenAI

A Python application that extracts news articles from URLs, generates summaries and identifies topics using GenAI tools, and enables semantic search with ChromaDB. Features a user-friendly Streamlit web interface for easy interaction.

## Features

- Extract headlines and full text from news article URLs
- Generate concise or detailed summaries using LangChain and OpenAI
- Identify main topics and keywords from articles
- Store articles, summaries, and topics in ChromaDB vector database
- Perform semantic search based on natural language queries
- Interactive Streamlit web interface for easy interaction
- Command-line interface for scripted operations

## Architecture

The application consists of the following components:

- **Article Extraction**: Extracts news article content from URLs using BeautifulSoup.
- **GenAI Summarization**: Uses LangChain and OpenAI to generate article summaries.
- **Topic Identification**: Uses LangChain and OpenAI to identify main topics and keywords.
- **Vector Database**: Stores articles with embeddings in ChromaDB for semantic search.
- **Semantic Search**: Enhanced search capabilities with query expansion using GenAI.
- **Web Interface**: Interactive Streamlit UI for user-friendly interaction.

## Setup

### Prerequisites

- Python 3.8 or higher
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

4. ChromaDB will use local persistent storage in the `data/chroma` directory.

### Usage

#### Web Interface (Recommended)

To run the Streamlit web interface:

```bash
poetry run streamlit run app.py
```

This will launch a web browser with an interactive interface where you can:
1. Add news articles by entering URLs
2. Browse all stored articles with summaries and topics
3. Search articles using natural language queries with semantic search

#### Command Line Interface

To run the application with example articles from the command line:

```bash
poetry run python main.py
```

This will:
1. Extract content from example news articles
2. Generate summaries and identify topics
3. Store the articles in ChromaDB
4. Demonstrate semantic search capabilities in the terminal

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
│   ├── search.py         # Semantic search functionality
│   └── ui.py             # Streamlit UI components
├── app.py                # Streamlit application entry point
├── main.py               # CLI application entry point
├── tests/                # Unit tests
├── data/                 # Data storage (ChromaDB files)
│   └── chroma/           # Persistent ChromaDB storage
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