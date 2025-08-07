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
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

pc = Pinecone(api_key=pinecone_config.api_key)

index = pc.Index(pinecone_config.index_name)

# Initialize vectorstores only if indexes exist
vectorstore = PineconeVectorStore(index=index, embedding=embeddings) if index else None


@tool
def search_google(query: str) -> str:
    """Performs a Google search using the provided query and returns the results."""
    try:
        results = search.run(query)
        return results
    except Exception as e:
        logger.error(f"Google search error: {str(e)}")
        return f"Error performing Google search: {str(e)}"



@tool
def search_knowledge_base(query: str) -> dict:
    """
    Searches internal knowledge base for company documentation, policies, and expertise.

    Args:
        query: Search query
    Returns:
        Dictionary with content and file names separately. like this
        {
            "content": content,
            "files": file_names
        }
    """
    try:
        if not vectorstore:
            return {"content": "Vector database not available. Please check Pinecone configuration.", "files": []}
            
        results = vectorstore.similarity_search(query, k=3)

        content_parts = []
        file_names = []
        
        for i, doc in enumerate(results, 1):
            file_name = doc.metadata.get('file_name', 'Unknown file')
            content_parts.append(f"Document {i}:\n{doc.page_content}")
            file_names.append(file_name)

        content = "\n\n".join(content_parts) if content_parts else "No relevant documents found."
        
        return {
            "content": content,
            "files": file_names
        }
    except Exception as e:
        logger.error(f"Knowledge base search error: {str(e)}")
        return {"content": f"Error searching knowledge base: {str(e)}", "files": []}



@tool
def market_research_agent(query: str) -> str:
    """
    Market Research Analyst persona.
    Handles market analysis, competitive research, customer insights, and industry trends.
    Uses web search for latest market data and vector DB for competitive intelligence.

    Returns:
        Comprehensive market analysis and insights.
    """

    system_prompt = """You are a Senior Market Research Analyst with 15+ years experience. 
    You specialize in market sizing, competitive analysis, customer behavior, and industry trends.
    Provide data-driven insights with specific metrics and actionable recommendations."""


    messages = [
        ("system", system_prompt),
        ("user", f"Market Research Question: {query}")
    ]

    response = llm.invoke(messages)
    return response.content


@tool
def technical_architect_agent(query: str) -> str:
    """
    Technical Architecture persona.
    Handles system design, technology decisions, implementation feasibility, and technical risk assessment.
    Uses knowledge base for internal tech stack and web search for latest technologies.

    Returns:
        Technical analysis and architecture recommendations.
    """

    system_prompt = """You are a Senior Technical Architect with expertise in system design, 
    cloud architecture, and technology strategy. You focus on scalability, security, 
    maintainability, and implementation feasibility."""

    messages = [
        ("system", system_prompt),
        ("user", f"Technical Question: {query}")
    ]

    response = llm.invoke(messages)
    return response.content


@tool
def financial_analyst_agent(query: str) -> str:
    """
    Financial Analysis persona.
    Handles ROI calculations, cost-benefit analysis, budget planning, and financial projections.
    Uses knowledge base for internal financial data and use cases for similar project costs.

    Returns:
        Detailed financial analysis and projections.
    """

    system_prompt = """You are a Senior Financial Analyst specializing in technology investments,
    project ROI analysis, and strategic financial planning. You provide detailed cost breakdowns,
    risk-adjusted returns, and clear financial recommendations."""


    messages = [
        ("system", system_prompt),
        ("user", f"Financial Analysis Question: {query}")
    ]

    response = llm.invoke(messages)
    return response.content


@tool
def risk_assessment_agent(query: str) -> str:
    """
    Risk Assessment Specialist persona.
    Identifies potential risks, compliance issues, regulatory requirements, and mitigation strategies.
    Uses web search for latest regulations and knowledge base for internal risk policies.

    Returns:
        Comprehensive risk analysis and mitigation recommendations.
    """


    system_prompt = """You are a Senior Risk Management Specialist with expertise in 
    regulatory compliance, operational risk, strategic risk assessment, and mitigation planning.
    You identify potential risks and provide practical mitigation strategies."""


    messages = [
        ("system", system_prompt),
        ("user", f"Risk Assessment Question: {query}")
    ]

    response = llm.invoke(messages)
    return response.content


@tool
def data_scientist_agent(query: str) -> str:
    """
    Data Science persona.
    Handles statistical analysis, predictive modeling, data interpretation, and quantitative insights.
    Uses knowledge base for internal data and use cases for similar analytical approaches.

    Returns:
        Data-driven insights and analytical recommendations.
    """

    system_prompt = """You are a Senior Data Scientist with expertise in statistical analysis,
    machine learning, predictive modeling, and business intelligence. You provide quantitative
    insights and data-driven recommendations."""

    messages = [
        ("system", system_prompt),
        ("user", f"Data Science Question: {query}")
    ]

    response = llm.invoke(messages)
    return response.content


@tool
def strategy_orchestrator(query: str) -> str:
    """
    Senior Strategy Persona - The Conductor.
    Breaks down complex questions, coordinates specialist personas, and synthesizes responses.
    This is the main entry point that orchestrates the entire workflow.

    Returns:
        Comprehensive strategic analysis combining insights from all relevant personas.
    """
    system_prompt = """You are a Senior Strategy Consultant and the orchestrator of a team of AI specialists.
    
    Your role:
    1. Analyze the query and determine which specialist personas are needed
    2. Break down complex questions into focused sub-questions
    3. Coordinate with relevant specialists: market_research_agent, technical_architect_agent, 
       financial_analyst_agent, risk_assessment_agent, data_scientist_agent
    4. Synthesize all responses into a coherent, actionable strategic recommendation
    
    Always structure your final response with:
    - Executive Summary with clear recommendation
    - Key insights from each specialist
    - Risk assessment and mitigation
    - Implementation roadmap
    - Success metrics
    
    Be decisive, data-driven, and provide actionable next steps."""

    messages = [
        ("system", system_prompt),
        ("user", f"Strategic Question: {query}\n\nPlease orchestrate the analysis using relevant specialist personas and provide a comprehensive strategic recommendation.")
    ]

    response = llm.invoke(messages)
    return response.content
    