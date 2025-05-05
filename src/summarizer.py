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
            
            return ArticleSummary(
                original_article=article,
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
                    "Always Return only a JSON with two lists: 'topics' and 'keywords'."
                    "If no topics or keywords are found, return empty lists respectively."
                )),
                HumanMessage(content=full_text)
            ]
            
            # Get response from model
            response = self.model.invoke(messages)
            
            # Process response to extract topics and keywords
            result = self._parse_topics_response(str(response.content))
            
            logger.info(f"Successfully identified topics and keywords")
            
            return TopicIdentification(
                original_article=article,
                topics=result["topics"],
                keywords=result["keywords"]
            )
            
        except Exception as e:
            logger.error(f"Error identifying topics: {str(e)}")
            raise ValueError(f"Failed to identify topics: {str(e)}")
    
    def _create_concise_chain(self):
        """Create a chain for generating concise summaries."""
        prompt_template = """
        Write a concise summary of the following article in no more than 3-4 sentences:
        
        {text}
        
        CONCISE SUMMARY:
        """
        prompt = PromptTemplate.from_template(prompt_template)
        return load_summarize_chain(self.model, chain_type="stuff", prompt=prompt)
    
    def _create_detailed_chain(self):
        """Create a chain for generating detailed summaries."""
        prompt_template = """
        Write a comprehensive summary of the following article. Include the main points, key details, and conclusions:
        
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
            if "topics" not in result or "keywords" not in result:
                logger.warning("Response did not contain expected keys")
                result = {
                    "topics": result.get("topics", ["General News"]),
                    "keywords": result.get("keywords", ["news", "article"])
                }
            
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
                
                if current_list is not None and line and not line.startswith(("topic", "keyword")):
                    # Extract words or phrases
                    items = re.findall(r'"([^"]+)"|(\w+)', line)
                    for item in items:
                        item_text = item[0] if item[0] else item[1]
                        if item_text and len(item_text) > 1:
                            current_list.append(item_text)
            
            # Ensure we have at least something in each category
            if not topics:
                topics = ["General News"]
            if not keywords:
                keywords = ["news", "article"]
            
            return {
                "topics": topics[:5],  # Limit to 5 topics
                "keywords": keywords[:10]  # Limit to 10 keywords
            }
