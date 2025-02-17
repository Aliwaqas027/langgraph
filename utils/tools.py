from langchain_core.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from utils.config import google_config, pinecone_config, azure_config
import logging

logger = logging.getLogger(__name__)

# Initialize LLM
llm = OpenAI()

# Initialize Google Search
search = GoogleSearchAPIWrapper(
    google_api_key=google_config.api_key,
    google_cse_id=google_config.cse_id
)

# Initialize Pinecone with embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002text-ada-002")

pc = Pinecone(api_key=pinecone_config.api_key)
index = pc.Index(pinecone_config.index_name)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)


@tool(return_direct=True)
def search_google(query: str) -> str:
    """Performs a Google search using the provided query and returns the results."""
    try:
        results = search.run(query)
        return results
    except Exception as e:
        logger.error(f"Google search error: {str(e)}")
        return f"Error performing Google search: {str(e)}"


@tool(return_direct=True)
def search_knowledge_base(query: str) -> str:
    """Searches the knowledge base for relevant information based on the provided query."""
    try:
        results = vectorstore.similarity_search(query, k=2)

        if not results:
            return "No relevant information found in knowledge base."

        formatted_results = []
        for i, doc in enumerate(results, 1):
            formatted_results.append(f"Document {i}:\n{doc.page_content}\n")

        return "\n".join(formatted_results)

    except Exception as e:
        logger.error(f"Knowledge base search error: {str(e)}")
        return f"Error searching knowledge base: {str(e)}"


@tool(return_direct=True)
def combine_search_results(query: str) -> str:
    """Combines results from both Google search and knowledge base search."""
    try:
        google_results = search_google(query)
        kb_results = search_knowledge_base(query)

        combined = f"""
        Google Search Results:
        ---------------------
        {google_results}
        
        Knowledge Base Results:
        ---------------------
        {kb_results}
        """

        return combined

    except Exception as e:
        logger.error(f"Combined search error: {str(e)}")
        return f"Error combining search results: {str(e)}"


def format_search_results(results: list) -> str:
    """Formats a list of search results into a readable string."""
    try:
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"{i}. {result}")
        return "\n".join(formatted)
    except Exception as e:
        logger.error(f"Error formatting results: {str(e)}")
        return str(results)  # Return raw results if formatting fails
