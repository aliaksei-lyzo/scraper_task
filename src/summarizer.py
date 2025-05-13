"""
Article summarization module.

This module provides functionality to summarize news articles using LangChain and OpenAI.
"""

import logging
from typing import List, Dict
from pydantic import SecretStr

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import OpenAIConfig
from src.models import ArticleContent, ArticleSummary, TopicIdentification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArticleSummarizer:
    """
    Summarizes news articles using LangChain and OpenAI.
    
    This class provides methods to generate concise summaries and 
    identify topics from article content.
    """
    
    def __init__(self):
        """Initialize the ArticleSummarizer with OpenAI configuration."""
        # Validate OpenAI configuration
        OpenAIConfig.validate()
        
        # Initialize the ChatOpenAI model
        self.model = ChatOpenAI(
            model=OpenAIConfig.MODEL,
            temperature=OpenAIConfig.TEMPERATURE,
            api_key=SecretStr(OpenAIConfig.API_KEY),
        )
        logger.info(f"Initialized ArticleSummarizer with model: {OpenAIConfig.MODEL}")
        
        # Text splitter for handling long articles
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def summarize(self, article: ArticleContent, summary_type: str = "concise") -> ArticleSummary:
        """
        Generate a summary of the article content.
        
        Args:
            article: The article content to summarize
            summary_type: The type of summary to generate (concise or detailed)
            
        Returns:
            ArticleSummary: Contains the original article and its summary
            
        Raises:
            ValueError: If summarization fails
        """
        try:
            logger.info(f"Generating {summary_type} summary for article: {article.title}")
            
            # Combine title and text for context
            full_text = f"Title: {article.title}\n\n{article.text}"
            
            # Split text for long articles
            docs = self.text_splitter.create_documents([full_text])
            
            # Create summarization chain based on summary type
            if summary_type == "detailed":
                chain = self._create_detailed_chain()
            else:  # Default to concise
                chain = self._create_concise_chain()
            
            # Run the chain
            summary = chain.run(docs)
            
            logger.info(f"Successfully generated {summary_type} summary")
            article_id = str(hash(f"{article.url}-{article.title}"))
            
            return ArticleSummary(
                article_id=article_id,
                summary=summary,
                summary_type=summary_type
            )
            
        except Exception as e:
            logger.error(f"Error summarizing article: {str(e)}")
            raise ValueError(f"Failed to summarize article: {str(e)}")
    def identify_topics(self, article: ArticleContent) -> TopicIdentification:
        """
        Identify main topics and keywords from article content.
        
        Args:
            article: The article content to analyze
            
        Returns:
            TopicIdentification: Contains identified topics and keywords
            
        Raises:
            ValueError: If topic identification fails
        """
        try:
            logger.info(f"Identifying topics for article: {article.title}")
            
            # Combine title and text for context
            full_text = f"Title: {article.title}\n\n{article.text}"
            
            # Create messages for the chat model
            messages = [
                SystemMessage(content=(
                    "You are an expert at analyzing news articles and identifying main topics and keywords. "
                    "Identify the 3-5 main topics and 5-10 relevant keywords from the article. "
                    "Also identify a single overall classification for the article (e.g., 'politics', 'technology', 'sports', 'health', etc.). "
                    "Always return only a JSON with three keys: 'classification', 'topics', and 'keywords'. "
                    "The 'classification' should be a single string, and 'topics' and 'keywords' should be lists. "
                    "Add the classification as the first element in both the topics and keywords lists. "
                    "Example format: {'classification': 'gaming', 'topics': ['gaming', 'fortnite leaks', 'fortnite gameplay'], 'keywords': ['gaming', 'battle royale', 'skins', 'update']}. "
                    "If no topics or keywords are found, include just the classification in the lists."
                )),
                HumanMessage(content=full_text)
            ]
            
            # Get response from model
            response = self.model.invoke(messages)
            
            # Process response to extract topics and keywords
            result = self._parse_topics_response(str(response.content))
            
            logger.info(f"Successfully identified topics and keywords")
            article_id = str(hash(f"{article.url}-{article.title}"))
            return TopicIdentification(
                article_id=article_id,
                topics=result["topics"],
                keywords=result["keywords"]
            )
            
        except Exception as e:
            logger.error(f"Error identifying topics: {str(e)}")
            raise ValueError(f"Failed to identify topics: {str(e)}")
            
    def _create_concise_chain(self):
        """Create a chain for generating concise summaries."""
        prompt_template = """
        Write a concise summary of the following article in no more than 3-4 sentences.
        First, identify a single main classification category for the article (e.g., politics, technology, sports, health).
        Begin your summary with "Classification: <category> - " followed by your summary text.
        
        {text}
        
        CONCISE SUMMARY:
        """
        prompt = PromptTemplate.from_template(prompt_template)
        return load_summarize_chain(self.model, chain_type="stuff", prompt=prompt)
    def _create_detailed_chain(self):
        """Create a chain for generating detailed summaries."""
        prompt_template = """
        Write a comprehensive summary of the following article. Include the main points, key details, and conclusions.
        Include the article's main classification (e.g., politics, technology, sports, health) at the beginning of the summary in the format "Classification: <category> - <summary text>":
        
        {text}
        
        DETAILED SUMMARY:
        """
        prompt = PromptTemplate.from_template(prompt_template)
        return load_summarize_chain(self.model, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt)
    
    def _parse_topics_response(self, response_content: str) -> Dict[str, List[str]]:
        """
        Parse the response from the model to extract topics and keywords.
        
        Args:
            response_content: The content returned by the model
            
        Returns:
            Dict containing topics and keywords lists
        """
        import json
        import re
        
        # Try to extract JSON from the response
        try:
            # Check for JSON block in markdown
            match = re.search(r"```json\s*(.*?)\s*```", response_content, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # Try to find any JSON-like structure
                match = re.search(r"\{.*\}", response_content, re.DOTALL)
                if match:
                    json_str = match.group(0)
                else:
                    json_str = response_content
              # Parse the JSON
            result = json.loads(json_str)
            
            # Ensure expected keys are present
            classification = result.get("classification", "general")
            
            if "topics" not in result or "keywords" not in result:
                logger.warning("Response did not contain expected keys")
                result = {
                    "topics": [classification, "General News"],
                    "keywords": [classification, "news", "article"]
                }
            else:
                # Add classification as first element if it's not already there
                topics = result.get("topics", [])
                if not topics or topics[0] != classification:
                    topics.insert(0, classification)
                
                keywords = result.get("keywords", [])
                if not keywords or keywords[0] != classification:
                    keywords.insert(0, classification)
                    
                result["topics"] = topics
                result["keywords"] = keywords
            
            return result
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse JSON from response: {str(e)}")
            # Fallback to simple parsing
            lines = response_content.strip().split('\n')
            topics = []
            keywords = []
            
            current_list = None
            for line in lines:
                line = line.strip()
                if "topic" in line.lower():
                    current_list = topics
                    line = re.sub(r".*topics?:?", "", line, flags=re.IGNORECASE).strip()
                elif "keyword" in line.lower():
                    current_list = keywords
                    line = re.sub(r".*keywords?:?", "", line, flags=re.IGNORECASE).strip()
                
                if current_list is not None and line and not line.startswith(("topic", "keyword")):                    # Extract words or phrases
                    items = re.findall(r'"([^"]+)"|([^,;\s]+(?:\s+[^,;]+)*)', line)
                    for item in items:
                        item_text = item[0] if item[0] else item[1]
                        if item_text and len(item_text) > 1:
                            current_list.append(item_text.strip())# Determine overall classification from topics or keywords
            classification = "general"
            
            # See if we can extract classification from the text
            classification_match = re.search(r"classification[:\s]+([a-zA-Z]+)", response_content, re.IGNORECASE)
            if classification_match:
                classification = classification_match.group(1).lower()
            
            # Ensure we have at least something in each category
            if not topics:
                topics = [classification, "General News"]
            elif topics[0] != classification:
                topics.insert(0, classification)
                
            if not keywords:
                keywords = [classification, "news", "article"]
            elif keywords[0] != classification:
                keywords.insert(0, classification)
                
            # Limit to reasonable number of items
            topics = topics[:6]  # Classification + 5 topics max
            keywords = keywords[:11]  # Classification + 10 keywords max
            
            return {
                "topics": topics,
                "keywords": keywords
            }
