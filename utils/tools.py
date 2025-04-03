from langchain_core.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from utils.config import google_config, pinecone_config, azure_config
import logging

logger = logging.getLogger(__name__)

AGENT_MODEL = "gpt-4o"
llm = ChatOpenAI(
    model=AGENT_MODEL,
    temperature=0,
)

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
legal_store = PineconeVectorStore(index='legal', embedding=embeddings)
finance_store = PineconeVectorStore(index='finance', embedding=embeddings)


@tool
def search_google(query: str) -> str:
    """Performs a Google search using the provided query and returns the results."""
    try:
        results = search.run(query)
        return results
    except Exception as e:
        logger.error(f"Google search error: {str(e)}")
        return f"Error performing Google search: {str(e)}"


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


@tool
def frontend_agent_tool(query: str) -> str:
    """
    Processes frontend development queries.
    Used when the user asks about user interface design, client-side technologies, or frontend-specific implementation details.

    Returns:
       A detailed response with expert advice on frontend development.
    """
    messages = [("system", "You are a Frontend Development expert."), ("user", query)]
    response = llm.invoke(messages)
    return response.content


@tool
def backend_agent_tool(query: str) -> str:
    """
    Processes backend development queries.
    Used when the user asks about server-side programming, database management, or backend architecture.

    Returns:
       A detailed response with expert advice on backend development.
    """
    messages = [("system", "You are a Backend Development expert."), ("user", query)]
    response = llm.invoke(messages)
    return response.content


@tool
def designer_agent_tool(query: str) -> str:
    """
    Processes design-related queries.
    Used when the user asks about user experience, graphic design, or interface aesthetics.

    Returns:
       A detailed response with expert advice on design and user experience.
    """
    messages = [("system", "You are a Design expert."), ("user", query)]
    response = llm.invoke(messages)
    return response.content


@tool
def legal_expert(query: str) -> str:
    """
    Processes legal-related queries and give relevant answer from knowledge base.
    Used when the user asks about legal, terms, or clauses.

    Returns:
       A detailed answer from knowledge base.
    """
    try:
        results = legal_store.similarity_search(query, k=2)

        formatted_results = []
        for i, doc in enumerate(results, 1):
            formatted_results.append(f"Document {i}:\n{doc.page_content}\n")

        context = "\n".join(formatted_results)

        messages = [("system", "You are a legal expert."), ("user", f"My question: {query}. Relevant knowledge base: {context}.")]
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        logger.error(f"Knowledge base search error: {str(e)}")
        return f"Error searching knowledge base: {str(e)}"


@tool
def finance_expert(query: str) -> str:
    """
    Processes finance-related queries and give relevant answer from knowledge base.
    Used when the user asks about finance.

    Returns:
       A detailed answer from knowledge base.
    """
    try:
        results = finance_store.similarity_search(query, k=2)

        formatted_results = []
        for i, doc in enumerate(results, 1):
            formatted_results.append(f"Document {i}:\n{doc.page_content}\n")

        context = "\n".join(formatted_results)

        messages = [("system", "You are a finance expert."), ("user", f"My question: {query}. Relevant knowledge base: {context}.")]
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        logger.error(f"Knowledge base search error: {str(e)}")
        return f"Error searching knowledge base: {str(e)}"
