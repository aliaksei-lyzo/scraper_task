# News Scraper - Task List

## Task Example
- [x] Create initial project structure (2025-03-29) // Completed task
- [ ] Update README.md with setup and usage instructions (2025-03-29) // Uncompleted task


## Current Tasks
- [x] Set up Python virtual environment (2025-05-04) — 10 min
- [x] Create requirements.txt and install dependencies: LangChain, Streamlit, ChromaDB, pydantic, pytest (2025-05-04) — 10 min
- [x] Scaffold project folder structure (src/, tests/, etc.) (2025-05-04) — 10 min
- [x] Implement module to extract news article headline and full text from a given URL (2025-05-04) — 30 min
- [x] Add error handling and edge case support for extraction (2025-05-04) — 15 min
- [x] Integrate LangChain for GenAI summarization of article content (2025-05-05) — 30 min
- [x] Integrate LangChain for topic identification from article content (2025-05-05) — 20 min
- [x] Design and implement pydantic models for article, summary, and topic data (2025-05-05) — 15 min
- [x] Store articles, summaries, and topics in ChromaDB vector database (2025-05-04) — 20 min
  - [x] Add ChromaDB to project dependencies (2025-05-05) — 5 min
  - [x] Create database service module for ChromaDB interactions (2025-05-05) — 15 min
  - [x] Implement document storage functionality (2025-05-05) — 20 min
  - [x] Switch from Docker to local persistent ChromaDB (2025-05-06) — 15 min
- [x] Implement semantic search using GenAI and ChromaDB (2025-05-04) — 30 min
- [x] Build Streamlit UI: input URLs, display summaries, topics, and search results (2025-05-13) — 40 min
- [x] Write Pytest unit tests for extraction, summarization, topic identification, search, and UI (2025-05-13) — 30 min
- [x] Update README.md with setup, usage, and architecture details (2025-05-13) — 15 min

**TOTAL: 5h 35min**

## Upcoming Tasks
- [ ] Add support for additional GenAI models (optional)
- [ ] Enhance UI with advanced filtering and visualization (optional)

## Discovered During Work
- [ ] Dockerize ChromaDB for more consistent development and deployment (2025-05-05)
